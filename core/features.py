"""Croisement : fiche statique et table horaire d'un site, à partir des données en local.

Rien n'est téléchargé ici, sauf trois éléments légers et mis en cache : l'altitude
(IGN), la distance à la mer (trait de côte Natural Earth) et, sur demande, la
végétation Sentinel-2. Les autres sources se téléchargent dans leurs onglets.

Table horaire indexée en UTC :
- ref_* : station Météo-France de référence (mesures validées, codes 0 et 1 ;
  rayonnement aussi en code 9, presque toujours non validé) ;
- era5_* : ERA5-Land (ou ERA5) au point de grille le plus proche ;
- T_cible : température mesurée au site quand le site est une station (cible du modèle) ;
- heure_locale, jour_annee ; puis la fiche statique répétée sur chaque ligne à l'export.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import zones
from core.config import PROCESSED_DIR
from core.sources import bdnb, era5, landsat, sentinel2, terrain
from core.sources import cerema_lcz as lcz
from core.sources import meteofrance_stations as mfs

PROC = PROCESSED_DIR / "croisement"
PROC.mkdir(parents=True, exist_ok=True)

REF_VARS = {"T": (0, 1), "TD": (0, 1), "U": (0, 1), "FF": (0, 1), "DD": (0, 1), "GLO": (0, 1, 9), "N": (0, 1, 9)}
_IND_LABEL_TO_CODE = {v: k for k, v in lcz.INDICATEURS.items()}


def site_key(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def last_summer(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def era5_dataset(lat: float, lon: float) -> str | None:
    """Jeu ERA5 disponible en local pour ce site (ERA5-Land de préférence)."""
    return next((d for d in ("era5-land", "era5") if era5.is_ready(d, lat, lon)), None)


# --------------------------------------------------------------------------- #
# État des sources
# --------------------------------------------------------------------------- #
def status(site: dict, zone: str, radius: int, radius_sat: int, lcz_catalog) -> pd.DataFrame:
    lat, lon = site["lat"], site["lon"]
    rows = []

    def add(source, ok, detail, where):
        rows.append({"source": source, "état": "✅" if ok is True else ("–" if ok is None else "❌"),
                     "détail": detail, "où l'obtenir": where})

    per = mfs.downloaded_periods(zone)
    add("Stations Météo-France", bool(per), ", ".join(per) if per else "rien pour cette zone", "onglet Stations")
    ds = era5_dataset(lat, lon)
    if ds:
        e = era5.load(ds, lat, lon)
        add("ERA5", True, f"{ds}, {e.index.min():%d/%m/%Y} → {e.index.max():%d/%m/%Y}", "onglet ERA5")
    else:
        add("ERA5", False, "rien pour ce point", "onglet ERA5")
    terr = lcz.territory_at(lat, lon, lcz_catalog) if lcz_catalog is not None and len(lcz_catalog) else None
    if terr is None:
        add("LCZ Cerema", None, "site hors des 93 territoires", "–")
    else:
        add("LCZ Cerema", terr in lcz.downloaded(), f"territoire {terr}", "onglet Urbain")
    if zones.is_outre_mer(zone):
        add("BDNB", None, "métropole uniquement", "–")
    else:
        add("BDNB", bdnb.is_ready(lat, lon, radius), f"rayon {radius} m", "onglet Urbain")
    add("Landsat (LST)", landsat.is_ready(lat, lon, radius_sat), f"rayon {radius_sat} m", "onglet Satellite")
    y = last_summer()
    add("Sentinel-2 (végétation)", sentinel2.cache_path(lat, lon, radius, y).exists(),
        f"été {y}, rayon {radius} m", "bouton ci-dessous")
    add("Altitude, distance à la mer", True, "calculées automatiquement", "–")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fiche statique
# --------------------------------------------------------------------------- #
def static_features(site: dict, zone: str, radius: int, radius_sat: int, lcz_catalog,
                    lcz_loader=lcz.load) -> tuple[dict, list[str]]:
    """Variables fixes du site (NaN si la source manque) et liste des avertissements."""
    lat, lon = site["lat"], site["lon"]
    f: dict = {"lat": lat, "lon": lon, "rayon_m": radius, "rayon_sat_m": radius_sat}
    warn: list[str] = []

    for name, fn in (("altitude_m", terrain.altitude), ("dist_mer_km", terrain.distance_mer_km)):
        try:
            f[name] = fn(lat, lon)
        except Exception as exc:
            f[name] = np.nan
            warn.append(f"{name} : {exc}")

    # LCZ : classe au site, parts de chaque classe et indicateurs moyens dans le rayon
    f["lcz_site"] = None
    for code in lcz.LCZ_CLASSES:
        f[f"lcz_part_{code}"] = np.nan
    for code in lcz.INDICATEURS:
        f[f"lcz_{code}"] = np.nan
    terr = lcz.territory_at(lat, lon, lcz_catalog) if lcz_catalog is not None and len(lcz_catalog) else None
    if terr is not None and terr in lcz.downloaded():
        prof = lcz.site_profile(lcz_loader(terr), lat, lon, radius)
        f["lcz_site"] = prof["classe"]
        parts = dict(zip(prof["composition"].get("lcz_code", []), prof["composition"].get("part", [])))
        for code in lcz.LCZ_CLASSES:
            f[f"lcz_part_{code}"] = float(parts.get(code, 0.0)) if parts else np.nan
        for label, value in prof["indicateurs"].items():
            f[f"lcz_{_IND_LABEL_TO_CODE.get(label, label)}"] = float(value)

    # BDNB : forme urbaine
    for k in ("bdnb_n", "bdnb_hauteur_m", "bdnb_emprise", "bdnb_facades", "bdnb_annee_med",
              "bdnb_part_residentiel", "bdnb_part_tertiaire"):
        f[k] = np.nan
    if not zones.is_outre_mer(zone) and bdnb.is_ready(lat, lon, radius):
        b = bdnb.load(lat, lon, radius)
        m = bdnb.morphology(b, lat, lon, radius)
        u = m["usages_emprise"]
        f.update({"bdnb_n": m["n_batiments"], "bdnb_hauteur_m": m["bldheight"], "bdnb_emprise": m["blddensity"],
                  "bdnb_facades": m["vertohor"],
                  "bdnb_annee_med": float(b["annee_construction"].median()) if len(b) else np.nan,
                  "bdnb_part_residentiel": float(sum(v for k, v in u.items() if k.startswith("Résidentiel"))),
                  "bdnb_part_tertiaire": float(u.get("Tertiaire", 0.0))})

    # Landsat : température de surface d'été
    for k in ("lst_site", "lst_zone", "lst_ecart", "lst_rang", "lst_dates"):
        f[k] = np.nan
    if landsat.is_ready(lat, lon, radius_sat):
        res = landsat.load(lat, lon, radius_sat)
        if res["composite"] is not None:
            s = landsat.summary(res)
            f.update({"lst_site": s["site"], "lst_zone": s["zone"], "lst_ecart": s["ecart_median"],
                      "lst_rang": s["rang"], "lst_dates": s["dates"]})

    # Sentinel-2 : végétation (seulement si déjà calculée)
    for k in ("s2_arbres", "s2_herbe", "s2_ndvi"):
        f[k] = np.nan
    cache = sentinel2.cache_path(lat, lon, radius, last_summer())
    if cache.exists():
        v = json.loads(cache.read_text())
        f.update({"s2_arbres": v["treecover"], "s2_herbe": v["grasscover"], "s2_ndvi": v["ndvi_median"]})
    return f, warn


# --------------------------------------------------------------------------- #
# Table horaire
# --------------------------------------------------------------------------- #
def _station_utc(s: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Horodatage en UTC : les fichiers d'outre-mer sont en heure locale (normale)."""
    if zones.is_outre_mer(zone):
        s = s.copy()
        s.index = s.index - pd.Timedelta(hours=zones.utc_offset(zone))
    return s


def hourly_table(site: dict, zone: str, obs: pd.DataFrame | None, ref_station: str | None,
                 target_station: str | None) -> pd.DataFrame:
    lat, lon = site["lat"], site["lon"]
    parts = []
    if obs is not None and ref_station:
        s = _station_utc(obs[obs["NUM_POSTE"] == ref_station].set_index("time").sort_index(), zone)
        parts.append(pd.DataFrame({f"ref_{v}": mfs.filter_quality(s, v, codes)
                                   for v, codes in REF_VARS.items() if v in s}))
    ds = era5_dataset(lat, lon)
    if ds:
        parts.append(era5.load(ds, lat, lon).add_prefix("era5_"))
    if obs is not None and target_station:
        s = _station_utc(obs[obs["NUM_POSTE"] == target_station].set_index("time").sort_index(), zone)
        parts.append(mfs.filter_quality(s, "T").rename("T_cible").to_frame())
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, axis=1).sort_index()
    table = table[~table.index.duplicated()]
    drivers = [c for c in ("ref_T", "era5_T") if c in table]
    if drivers:
        table = table[table[drivers].notna().any(axis=1)]
    local = table.index + pd.Timedelta(hours=zones.utc_offset(zone))
    table["heure_locale"] = local.hour
    table["jour_annee"] = local.dayofyear
    table.index.name = "time_utc"
    return table


def diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    """Erreur de chaque prédicteur « naïf » contre la cible (si le site est une station)."""
    if "T_cible" not in table:
        return pd.DataFrame()
    rows = []
    for col, name in (("ref_T", "station de référence"), ("era5_T", "ERA5")):
        if col in table:
            both = table[["T_cible", col]].dropna()
            if len(both) >= 24:
                e = both[col] - both["T_cible"]
                rows.append({"prédicteur": name, "heures": len(both), "biais (°C)": e.mean(),
                             "MAE (°C)": e.abs().mean(), "RMSE (°C)": float(np.sqrt((e ** 2).mean()))})
    return pd.DataFrame(rows).round(2)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def save(site: dict, fiche: dict, table: pd.DataFrame) -> Path:
    key = site_key(site["lat"], site["lon"])
    (PROC / f"{key}_fiche.json").write_text(json.dumps({**fiche, "label": site.get("label")},
                                                        ensure_ascii=False, default=float))
    out = PROC / f"{key}_table.parquet"
    table.to_parquet(out)
    return out


def with_static(table: pd.DataFrame, fiche: dict) -> pd.DataFrame:
    """Table horaire + fiche statique répétée sur chaque ligne (format d'entrée du modèle)."""
    return table.assign(**{k: v for k, v in fiche.items() if k not in ("lat", "lon")})
