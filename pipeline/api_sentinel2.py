"""Sentinel-2 L2A – végétation autour du site (NDVI 10 m) via Microsoft Planetary Computer, sans clé.

On prend la scène d'été (juin-août) la moins nuageuse, on calcule le NDVI = (B08 − B04) / (B08 + B04)
sur les pixels sans nuage ni ombre (couche SCL), dans un disque autour du site.
Depuis la version de traitement 04.00 (janvier 2022), réflectance = (valeur − 1000) / 10 000.
Seuils indicatifs : arbres NDVI ≥ 0,5, herbe 0,3 à 0,5. Environ 10 s par site.

Exemple :
    python api_sentinel2.py 43.3048 5.3955
"""
from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point

import config

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RES = 10.0
SCL_OK = [2, 4, 5, 6, 7]     # ombre sombre, végétation, sol nu, eau, non classé
DOSSIER = config.dossier("sentinel2")


def caracteristiques(lat: float, lon: float, rayon_m: int, annee: int = config.ETE_SATELLITE) -> dict:
    """s2_arbres, s2_herbe (parts du disque), s2_ndvi (médiane). En cache."""
    cache = DOSSIER / f"ndvi_{lat:.4f}_{lon:.4f}_r{int(rayon_m)}_{annee}.json"
    if cache.exists():
        v = json.loads(cache.read_text())
        return {"s2_arbres": v["treecover"], "s2_herbe": v["grasscover"], "s2_ndvi": v["ndvi_median"]}
    client = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    d = rayon_m / 111_000 + 0.002
    scenes = list(client.search(collections=["sentinel-2-l2a"], bbox=[lon - d, lat - d, lon + d, lat + d],
                                datetime=f"{annee}-06-01/{annee}-08-31").items())
    if not scenes:
        raise ValueError(f"aucune scène Sentinel-2 à l'été {annee}")
    s = min(scenes, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    p = pt.to_crs(crs).iloc[0]
    n = int(np.ceil(rayon_m / RES))
    taille = 2 * n + 1
    tr = from_origin(p.x - (n + 0.5) * RES, p.y + (n + 0.5) * RES, RES, RES)

    def lire(bande: str) -> np.ndarray:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(s.assets[bande].href) as src, \
                WarpedVRT(src, crs=crs, transform=tr, width=taille, height=taille,
                          resampling=Resampling.nearest, nodata=0) as vrt:
            return vrt.read(1).astype("float32")

    decalage = 1000.0 if float(s.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    rouge, pir, scl = lire("B04") - decalage, lire("B08") - decalage, lire("SCL")
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    disque = (xx ** 2 + yy ** 2) * RES ** 2 <= rayon_m ** 2
    ok = disque & np.isin(scl, SCL_OK) & (rouge + pir > 0)
    if not ok.any():
        raise ValueError(f"scène {s.id} inexploitable sur la zone")
    ndvi = ((pir - rouge) / np.where(rouge + pir > 0, rouge + pir, 1))[ok]
    v = {"scene": s.id, "date": s.datetime.date().isoformat(), "part_pixels_valides": float(ok.sum() / disque.sum()),
         "treecover": float((ndvi >= 0.5).mean()), "grasscover": float(((ndvi >= 0.3) & (ndvi < 0.5)).mean()),
         "ndvi_median": float(np.median(ndvi))}
    cache.write_text(json.dumps(v))
    return {"s2_arbres": v["treecover"], "s2_herbe": v["grasscover"], "s2_ndvi": v["ndvi_median"]}


if __name__ == "__main__":
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (43.3048, 5.3955)
    print(caracteristiques(lat, lon, config.RAYON_URBAIN_M))
