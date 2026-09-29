"""BDNB – Base de données nationale des bâtiments (CSTB), API ouverte.

Chaîne : bâtiments dans un rayon autour d'un site (requête par emprise en
Lambert-93, filtrée ensuite au disque) -> GeoParquet par site -> lecture.

API sans clé : 10 lignes par requête et 120 requêtes par minute, d'où un
téléchargement par site plutôt que par commune. Métropole uniquement (aucun
bâtiment d'outre-mer dans l'API ouverte, vérifié sur 971 et 974).
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

from core.config import BDNB_API, PROCESSED_DIR
from core.net import get_json

PROC = PROCESSED_DIR / "bdnb"
PROC.mkdir(parents=True, exist_ok=True)

PAGE = 10           # plafond de l'API sans clé
PAUSE_S = 0.5       # 120 requêtes / minute
CRS = "EPSG:2154"   # projection des géométries renvoyées

# Colonnes gardées (utiles pour le confort d'été ; la table en compte ~140)
COLUMNS = {
    "batiment_groupe_id": "Identifiant BDNB",
    "libelle_adr_principale_ban": "Adresse",
    "code_commune_insee": "Commune (INSEE)",
    "usage_principal_bdnb_open": "Usage principal",
    "annee_construction": "Année de construction",
    "hauteur_mean": "Hauteur moyenne (m)",
    "nb_niveau": "Niveaux",
    "surface_emprise_sol": "Emprise au sol (m²)",
    "nb_log": "Logements",
    "mat_mur_txt": "Matériau des murs",
    "mat_toit_txt": "Matériau du toit",
    "classe_bilan_dpe": "Classe DPE",
    "classe_inertie": "Inertie thermique (DPE)",
    "type_isolation_mur_exterieur": "Isolation des murs (DPE)",
    "type_isolation_plancher_haut": "Isolation du plancher haut (DPE)",
    "type_generateur_climatisation": "Climatisation (DPE)",
    "pourcentage_surface_baie_vitree_exterieur": "Part vitrée des façades (DPE)",
    "facteur_solaire_baie_vitree": "Facteur solaire des vitrages (DPE)",
    "type_fermeture": "Protections solaires (DPE)",
    "traversant": "Logement traversant (DPE)",
    "altitude_sol_mean": "Altitude du sol (m)",
    "fiabilite_hauteur": "Fiabilité de la hauteur",
}


def site_path(lat: float, lon: float, radius_m: int) -> Path:
    return PROC / f"bdnb_{lat:.5f}_{lon:.5f}_r{int(radius_m)}.parquet"


def fetch(lat: float, lon: float, radius_m: int, progress=None) -> Path:
    """Télécharge les bâtiments dans le rayon ; progress(fraction) est optionnel."""
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS).iloc[0]
    xmin, ymin, xmax, ymax = site.buffer(radius_m).bounds
    params = {"xmin": round(xmin), "ymin": round(ymin), "xmax": round(xmax), "ymax": round(ymax),
              "select": ",".join(list(COLUMNS) + ["geom_groupe"]), "limit": PAGE}
    rows: list[dict] = []
    total = None
    offset = 0
    while total is None or offset < total:
        r = get_json(f"{BDNB_API}/bbox", params={**params, "offset": offset},
                     headers={"Prefer": "count=exact"} if total is None else None)
        batch = r.json()
        if total is None:  # « content-range: 0-9/691 »
            total = int(r.headers.get("content-range", "*/0").split("/")[-1] or 0)
        if not batch:
            break
        rows += batch
        offset += len(batch)
        if progress and total:
            progress(min(offset / total, 1.0))
        time.sleep(PAUSE_S)

    geoms = [shape(row.pop("geom_groupe")) if row.get("geom_groupe") else None for row in rows]
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows, columns=list(COLUMNS)), geometry=geoms, crs=CRS)
    gdf = gdf[gdf.geometry.notna()]
    gdf["distance_m"] = gdf.geometry.distance(site).round(1)
    gdf = gdf[gdf["distance_m"] <= radius_m].sort_values("distance_m").reset_index(drop=True)
    out = site_path(lat, lon, radius_m)
    gdf.to_parquet(out, index=False)
    return out


def is_ready(lat: float, lon: float, radius_m: int) -> bool:
    return site_path(lat, lon, radius_m).exists()


def load(lat: float, lon: float, radius_m: int) -> gpd.GeoDataFrame:
    return gpd.read_parquet(site_path(lat, lon, radius_m))


def summary(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_m: int) -> dict:
    """Quelques chiffres sur l'environnement bâti du site (emprises découpées au disque)."""
    disk = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS).iloc[0].buffer(radius_m)
    emprise = gdf.geometry.intersection(disk).area
    h = gdf["hauteur_mean"]
    annee = gdf["annee_construction"]
    return {
        "n": len(gdf),
        "part_batie": float(emprise.sum() / disk.area),
        "hauteur_moy": float((h * emprise).sum() / emprise[h.notna()].sum()) if h.notna().any() else None,
        "annee_med": float(annee.median()) if annee.notna().any() else None,
    }


def morphology(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_m: int) -> dict:
    """Paramètres de forme urbaine (ceux d'UWG) sur les bâtiments dont le centre est dans le disque.

    bldheight : hauteur moyenne pondérée par l'emprise (m) ; blddensity : emprise / surface du
    disque ; vertohor : surface de façades (périmètre x hauteur) / surface du disque.
    """
    disk = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS).iloc[0].buffer(radius_m)
    b = gdf.to_crs(CRS)
    b = b[b.geometry.centroid.within(disk)].copy()
    if b.empty:
        return {"n_batiments": 0, "bldheight": np.nan, "blddensity": 0.0, "vertohor": 0.0,
                "part_hauteur_manquante": np.nan, "usages_emprise": {}}
    emprise, perimetre = b.geometry.area, b.geometry.length
    h = b["hauteur_mean"].astype(float)
    hf = h.fillna(h.median()) if h.notna().any() else h
    usage = emprise.groupby(b["usage_principal_bdnb_open"].fillna("inconnu")).sum()
    return {
        "n_batiments": int(len(b)),
        "bldheight": float((hf * emprise).sum() / emprise[hf.notna()].sum()) if hf.notna().any() else np.nan,
        "blddensity": float(emprise.sum() / disk.area),
        "vertohor": float((perimetre * hf).sum() / disk.area) if hf.notna().any() else np.nan,
        "part_hauteur_manquante": float(h.isna().mean()),
        "usages_emprise": (usage / usage.sum()).round(3).to_dict(),
    }
