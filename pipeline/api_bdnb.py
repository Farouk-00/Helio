"""BDNB – Base de données nationale des bâtiments (CSTB), API ouverte sans clé.

Limites sans clé : 10 lignes par requête, 120 requêtes par minute. On demande donc
les bâtiments d'une emprise (Lambert-93) autour du site, page par page.
Métropole uniquement (aucun bâtiment d'outre-mer dans l'API ouverte).
Quelques secondes à une minute par site selon la densité.

Exemple :
    python api_bdnb.py 43.3048 5.3955
"""
from __future__ import annotations

import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

import config
from outils import get

API = "https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet"
COLONNES = ["batiment_groupe_id", "usage_principal_bdnb_open", "annee_construction",
            "hauteur_mean", "nb_niveau", "surface_emprise_sol", "mat_mur_txt", "mat_toit_txt",
            "classe_inertie"]
LAMBERT93 = "EPSG:2154"
DOSSIER = config.dossier("bdnb")


def chemin(lat: float, lon: float, rayon_m: int):
    return DOSSIER / f"bdnb_{lat:.5f}_{lon:.5f}_r{int(rayon_m)}.parquet"


def telecharger(lat: float, lon: float, rayon_m: int) -> gpd.GeoDataFrame:
    """Bâtiments dont l'emprise touche le carré englobant le disque (en cache)."""
    out = chemin(lat, lon, rayon_m)
    if out.exists():
        return gpd.read_parquet(out)
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(LAMBERT93).iloc[0]
    xmin, ymin, xmax, ymax = site.buffer(rayon_m).bounds
    params = {"xmin": round(xmin), "ymin": round(ymin), "xmax": round(xmax), "ymax": round(ymax),
              "select": ",".join(COLONNES + ["geom_groupe"]), "limit": 10}
    lignes, total = [], None
    while total is None or len(lignes) < total:
        r = get(f"{API}/bbox", params={**params, "offset": len(lignes)},
                headers={"Prefer": "count=exact"} if total is None else None)
        if total is None:                  # en-tête « content-range: 0-9/691 »
            total = int(r.headers.get("content-range", "*/0").split("/")[-1] or 0)
        page = r.json()
        if not page:
            break
        lignes += page
        time.sleep(0.5)                    # 120 requêtes / minute
    geoms = [shape(x.pop("geom_groupe")) if x.get("geom_groupe") else None for x in lignes]
    g = gpd.GeoDataFrame(pd.DataFrame(lignes, columns=COLONNES), geometry=geoms, crs=LAMBERT93)
    g = g[g.geometry.notna()]
    g.to_parquet(out, index=False)
    return g


def caracteristiques(lat: float, lon: float, rayon_m: int) -> dict:
    """Forme urbaine sur les bâtiments dont le centre est dans le disque.

    bdnb_hauteur_m : hauteur moyenne pondérée par l'emprise ; bdnb_emprise : part du disque
    couverte de bâti ; bdnb_facades : surface de façades / surface du disque ;
    bdnb_annee_med : année de construction médiane ; bdnb_part_residentiel / _tertiaire.
    """
    vide = {"bdnb_n": np.nan, "bdnb_hauteur_m": np.nan, "bdnb_emprise": np.nan, "bdnb_facades": np.nan,
            "bdnb_annee_med": np.nan, "bdnb_part_residentiel": np.nan, "bdnb_part_tertiaire": np.nan}
    g = telecharger(lat, lon, rayon_m).to_crs(LAMBERT93)
    disque = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(LAMBERT93).iloc[0].buffer(rayon_m)
    b = g[g.geometry.centroid.within(disque)]
    if b.empty:
        return {**vide, "bdnb_n": 0, "bdnb_emprise": 0.0, "bdnb_facades": 0.0}
    emprise, perimetre = b.geometry.area, b.geometry.length
    h = pd.to_numeric(b["hauteur_mean"], errors="coerce")
    h = h.fillna(h.median())
    usage = emprise.groupby(b["usage_principal_bdnb_open"].fillna("inconnu")).sum() / emprise.sum()
    return {
        "bdnb_n": len(b),
        "bdnb_hauteur_m": float((h * emprise).sum() / emprise[h.notna()].sum()) if h.notna().any() else np.nan,
        "bdnb_emprise": float(emprise.sum() / disque.area),
        "bdnb_facades": float((perimetre * h).sum() / disque.area) if h.notna().any() else np.nan,
        "bdnb_annee_med": float(pd.to_numeric(b["annee_construction"], errors="coerce").median()),
        "bdnb_part_residentiel": float(sum(v for k, v in usage.items() if str(k).startswith("Résidentiel"))),
        "bdnb_part_tertiaire": float(usage.get("Tertiaire", 0.0)),
    }


if __name__ == "__main__":
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (43.3048, 5.3955)
    print(caracteristiques(lat, lon, config.RAYON_URBAIN_M))
