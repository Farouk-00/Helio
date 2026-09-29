"""Sentinel-2 L2A – végétation autour d'un site (NDVI 10 m), via Planetary Computer.

Scène d'été la moins nuageuse sur la zone ; parts d'arbres (NDVI >= 0,5) et
d'herbe (0,3 <= NDVI < 0,5) dans un disque. Seuils indicatifs, non calés.
Résultat mis en cache (JSON) par site et par rayon.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point

from core.config import PC_STAC_URL, PROCESSED_DIR

PROC = PROCESSED_DIR / "sentinel2"
PROC.mkdir(parents=True, exist_ok=True)
RES = 10.0
SCL_OK = [2, 4, 5, 6, 7]   # ombre sombre, végétation, sol nu, eau, non classé (hors nuages / ombres)
TREE, GRASS = 0.5, 0.3


def cache_path(lat: float, lon: float, radius_m: int, year: int) -> Path:
    return PROC / f"ndvi_{lat:.4f}_{lon:.4f}_r{int(radius_m)}_{year}.json"


def vegetation(lat: float, lon: float, radius_m: int, year: int, refresh: bool = False) -> dict:
    """{scene, nuages_scene, part_pixels_valides, treecover, grasscover, ndvi_median} (été = juin-août)."""
    cache = cache_path(lat, lon, radius_m, year)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    client = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    d = radius_m / 111_000 + 0.002
    items = list(client.search(collections=["sentinel-2-l2a"], bbox=[lon - d, lat - d, lon + d, lat + d],
                               datetime=f"{year}-06-01/{year}-08-31").items())
    if not items:
        raise ValueError(f"aucune scène Sentinel-2 à l'été {year}")
    it = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    p = pt.to_crs(crs).iloc[0]
    n = int(np.ceil(radius_m / RES))
    size = 2 * n + 1
    transform = from_origin(p.x - (n + 0.5) * RES, p.y + (n + 0.5) * RES, RES, RES)

    def read(asset: str) -> np.ndarray:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(it.assets[asset].href) as src, \
                WarpedVRT(src, crs=crs, transform=transform, width=size, height=size,
                          resampling=Resampling.nearest, nodata=0) as vrt:
            return vrt.read(1).astype("float32")

    # Depuis la version de traitement 04.00 (janv. 2022) : réflectance = (CN - 1000) / 10 000
    offset = 1000.0 if float(it.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    red, nir, scl = read("B04") - offset, read("B08") - offset, read("SCL")
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    disk = (xx ** 2 + yy ** 2) * RES ** 2 <= radius_m ** 2
    ok = disk & np.isin(scl, SCL_OK) & (red + nir > 0)
    if not ok.any():
        raise ValueError(f"scène {it.id} inexploitable sur la zone")
    ndvi = (nir - red) / np.where(red + nir > 0, red + nir, 1)
    v = ndvi[ok]
    res = {"scene": it.id, "date": it.datetime.date().isoformat(),
           "nuages_scene": float(it.properties.get("eo:cloud_cover", np.nan)),
           "part_pixels_valides": float(ok.sum() / disk.sum()),
           "treecover": float((v >= TREE).mean()), "grasscover": float(((v >= GRASS) & (v < TREE)).mean()),
           "ndvi_median": float(np.median(v))}
    cache.write_text(json.dumps(res))
    return res
