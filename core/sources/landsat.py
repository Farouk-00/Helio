"""Landsat 8/9 – température de surface (LST, Collection 2 niveau 2), Planetary Computer.

Chaîne : recherche STAC des scènes d'été sur la zone du site -> lecture de la
seule fenêtre utile (bande thermique `lwir11` + masque `qa_pixel`, COG distants)
reprojetée sur une grille UTM de 30 m centrée sur le site -> pile de scènes en
GeoTIFF + tableau par scène (Parquet) + composite médian.

LST = température de la surface (toits, chaussée, végétation), pas de l'air.
Passage vers 10 h 30 UTC, jamais de nuit.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point

from core.config import PC_STAC_URL, PROCESSED_DIR

PROC = PROCESSED_DIR / "landsat"
PROC.mkdir(parents=True, exist_ok=True)

COLLECTION = "landsat-c2-l2"
PLATFORMS = ("landsat-8", "landsat-9")
SCALE, OFFSET = 0.00341802, 149.0   # ST_B10 -> kelvins
RES = 30.0
# qa_pixel : bit 0 remplissage, 1 nuage dilaté, 2 cirrus, 3 nuage, 4 ombre de nuage, 5 neige
QA_REJECT = 0b111111
MIN_VALID = 0.3                     # part de pixels valides pour garder une date
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif",
            "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}


# --------------------------------------------------------------------------- #
# Grille du site
# --------------------------------------------------------------------------- #
def site_key(lat: float, lon: float, radius_m: int) -> str:
    return f"{lat:.4f}_{lon:.4f}_r{int(radius_m)}"


def site_dir(lat: float, lon: float, radius_m: int) -> Path:
    return PROC / site_key(lat, lon, radius_m)


def grid(lat: float, lon: float, radius_m: int):
    """Grille UTM de 30 m centrée sur le site : (crs, transform, taille, masque du disque)."""
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    p = pt.to_crs(crs).iloc[0]
    n = int(np.ceil(radius_m / RES))
    size = 2 * n + 1
    transform = from_origin(p.x - (n + 0.5) * RES, p.y + (n + 0.5) * RES, RES, RES)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    disk = (xx ** 2 + yy ** 2) * RES ** 2 <= radius_m ** 2
    return crs, transform, size, disk


# --------------------------------------------------------------------------- #
# Recherche et lecture
# --------------------------------------------------------------------------- #
def search(lat: float, lon: float, radius_m: int, years: list[int], months: list[int],
           max_cloud: float) -> list:
    """Scènes Landsat 8/9 des mois demandés, couverture nuageuse de la scène <= max_cloud %."""
    client = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    d = radius_m / 111_000 + 0.001
    bbox = [lon - d / np.cos(np.radians(lat)), lat - d, lon + d / np.cos(np.radians(lat)), lat + d]
    items = []
    for y in sorted(years):
        m1, m2 = min(months), max(months)
        end = (dt.date(y + (m2 == 12), m2 % 12 + 1, 1) - dt.timedelta(days=1)).isoformat()
        found = client.search(collections=[COLLECTION], bbox=bbox,
                              datetime=f"{y}-{m1:02d}-01/{end}").items()
        items += [it for it in found
                  if it.properties.get("platform") in PLATFORMS
                  and it.datetime.month in months
                  and (it.properties.get("eo:cloud_cover") or 0) <= max_cloud]
    return sorted(items, key=lambda it: it.datetime)


def _read(href: str, crs, transform, size: int, nodata: int) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=transform, width=size,
                                               height=size, resampling=Resampling.nearest,
                                               nodata=nodata) as vrt:
        return vrt.read(1)


def read_item(item, crs, transform, size: int) -> np.ndarray:
    """LST (°C) d'une scène sur la grille du site ; NaN hors scène, nuages, ombres, neige."""
    with rasterio.Env(**GDAL_ENV):
        raw = _read(item.assets["lwir11"].href, crs, transform, size, nodata=0)
        qa = _read(item.assets["qa_pixel"].href, crs, transform, size, nodata=1)
    valid = (raw > 0) & ((qa & QA_REJECT) == 0)
    return np.where(valid, raw * SCALE + OFFSET - 273.15, np.nan).astype("float32")


def _site_value(img: np.ndarray) -> float:
    """Médiane des 3 x 3 pixels (90 m) autour du site."""
    c = img.shape[0] // 2
    win = img[c - 1:c + 2, c - 1:c + 2]
    return float(np.nanmedian(win)) if np.isfinite(win).any() else np.nan


# --------------------------------------------------------------------------- #
# Téléchargement + stockage
# --------------------------------------------------------------------------- #
def fetch(lat: float, lon: float, radius_m: int, years: list[int], months: list[int],
          max_cloud: float = 30, progress=None) -> Path:
    """Lit toutes les scènes, les regroupe par date et écrit pile + tableau + composite."""
    crs, transform, size, disk = grid(lat, lon, radius_m)
    items = search(lat, lon, radius_m, years, months, max_cloud)
    by_date: dict[dt.date, dict] = {}
    for i, it in enumerate(items):
        img = read_item(it, crs, transform, size)
        day = it.datetime.date()
        if day in by_date:  # deux scènes voisines le même jour : on complète les trous
            prev = by_date[day]
            prev["img"] = np.where(np.isnan(prev["img"]), img, prev["img"])
            prev["ids"].append(it.id)
        else:
            by_date[day] = {"img": img, "ids": [it.id], "heure_utc": it.datetime.strftime("%H:%M"),
                            "plateforme": it.properties.get("platform"),
                            "nuages_scene": it.properties.get("eo:cloud_cover")}
        if progress:
            progress((i + 1) / max(len(items), 1))

    rows, keep = [], []
    for day, d in sorted(by_date.items()):
        img = d["img"]
        part = float(np.isfinite(img[disk]).mean())
        site, zone = _site_value(img), float(np.nanmean(img[disk])) if part > 0 else np.nan
        rows.append({"date": pd.Timestamp(day), "heure_utc": d["heure_utc"], "plateforme": d["plateforme"],
                     "nuages_scene": d["nuages_scene"], "part_valide": round(part, 3),
                     "lst_site": site, "lst_zone": zone, "ecart": site - zone,
                     "retenue": part >= MIN_VALID, "scenes": ",".join(d["ids"])})
        if part >= MIN_VALID:
            keep.append((day, img))

    out = site_dir(lat, lon, radius_m)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "heure_utc", "plateforme", "nuages_scene", "part_valide",
                                "lst_site", "lst_zone", "ecart", "retenue", "scenes"]
                 ).to_parquet(out / "scenes.parquet", index=False)
    profile = {"driver": "GTiff", "dtype": "float32", "crs": crs, "transform": transform,
               "width": size, "height": size, "nodata": np.nan, "compress": "deflate"}
    if keep:
        stack = np.stack([img for _, img in keep])
        with rasterio.open(out / "pile.tif", "w", count=len(keep), **profile) as dst:
            dst.write(stack)
            for b, (day, _) in enumerate(keep, start=1):
                dst.set_band_description(b, day.isoformat())
        with np.errstate(all="ignore"):
            comp = np.stack([np.nanmedian(stack, axis=0), np.isfinite(stack).sum(axis=0).astype("float32")])
        with rasterio.open(out / "composite.tif", "w", count=2, **profile) as dst:
            dst.write(comp)
            dst.set_band_description(1, "LST médiane (°C)")
            dst.set_band_description(2, "nombre de dates valides")
    (out / "meta.json").write_text(json.dumps({
        "lat": lat, "lon": lon, "rayon_m": radius_m, "annees": sorted(years), "mois": sorted(months),
        "nuages_max": max_cloud, "scenes": len(items), "dates_retenues": len(keep),
        "maj": dt.datetime.now().isoformat(timespec="seconds")}))
    return out


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def is_ready(lat: float, lon: float, radius_m: int) -> bool:
    return (site_dir(lat, lon, radius_m) / "meta.json").exists()


def load(lat: float, lon: float, radius_m: int) -> dict:
    """{meta, scenes, composite (°C), n_dates, crs, transform, disk} ; composite None si vide."""
    d = site_dir(lat, lon, radius_m)
    res = {"meta": json.loads((d / "meta.json").read_text()),
           "scenes": pd.read_parquet(d / "scenes.parquet"),
           "composite": None, "n_dates": None, "crs": None, "transform": None}
    if (d / "composite.tif").exists():
        with rasterio.open(d / "composite.tif") as src:
            res["composite"], res["n_dates"] = src.read(1), src.read(2)
            res["crs"], res["transform"] = src.crs, src.transform
    res["disk"] = grid(lat, lon, radius_m)[3]
    return res


def summary(res: dict) -> dict:
    """LST du site et de la zone sur le composite, écart médian site - zone par date."""
    comp, disk = res["composite"], res["disk"]
    kept = res["scenes"][res["scenes"]["retenue"]]
    zone = comp[disk]
    site = _site_value(comp)
    return {
        "site": site,
        "zone": float(np.nanmean(zone)),
        "ecart_median": float(kept["ecart"].median()) if kept["ecart"].notna().any() else np.nan,
        "rang": float((zone < site).sum() / np.isfinite(zone).sum()) if np.isfinite(site) else np.nan,
        "dates": len(kept),
    }
