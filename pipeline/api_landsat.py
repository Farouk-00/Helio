"""Landsat 8/9 – température de surface (LST, Collection 2 niveau 2) via Microsoft Planetary Computer.

Sans clé : catalogue STAC public, URLs signées automatiquement (paquet planetary-computer).
On ne lit que la petite fenêtre utile des images (COG distants), reprojetée sur une grille
de 30 m centrée sur le site. LST (°C) = lwir11 x 0,00341802 + 149 - 273,15.
Masque qa_pixel, bits 0-5 : remplissage, nuage dilaté, cirrus, nuage, ombre, neige.

LST = température de la surface (toits, routes, végétation) vers 10 h 30 UTC, pas de l'air,
jamais de nuit. Utile comme « signature » du site : est-il plus chaud que ses alentours ?
Environ 30 s à 1 min par site et par été.

Exemple :
    python api_landsat.py 43.3048 5.3955
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
RES = 30.0
DOSSIER = config.dossier("landsat")
GDAL = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif",
        "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}


def _grille(lat: float, lon: float, rayon_m: int):
    """Grille UTM de 30 m centrée sur le site : (crs, transform, taille, masque du disque)."""
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    p = pt.to_crs(crs).iloc[0]
    n = int(np.ceil(rayon_m / RES))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return crs, from_origin(p.x - (n + 0.5) * RES, p.y + (n + 0.5) * RES, RES, RES), 2 * n + 1, \
        (xx ** 2 + yy ** 2) * RES ** 2 <= rayon_m ** 2


def _lire(href: str, crs, tr, taille: int, nodata: int) -> np.ndarray:
    with rasterio.Env(**GDAL), rasterio.open(href) as src, \
            WarpedVRT(src, crs=crs, transform=tr, width=taille, height=taille,
                      resampling=Resampling.nearest, nodata=nodata) as vrt:
        return vrt.read(1)


def images_ete(lat: float, lon: float, rayon_m: int, annee: int, nuages_max: float = 30) -> list:
    """Images LST (°C, NaN si masqué) des scènes de juin à août de l'année."""
    crs, tr, taille, disque = _grille(lat, lon, rayon_m)
    client = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    d = rayon_m / 111_000 + 0.001
    bbox = [lon - d / np.cos(np.radians(lat)), lat - d, lon + d / np.cos(np.radians(lat)), lat + d]
    scenes = [s for s in client.search(collections=["landsat-c2-l2"], bbox=bbox,
                                       datetime=f"{annee}-06-01/{annee}-08-31").items()
              if s.properties.get("platform") in ("landsat-8", "landsat-9")
              and (s.properties.get("eo:cloud_cover") or 0) <= nuages_max]
    images = []
    for s in scenes:
        brut = _lire(s.assets["lwir11"].href, crs, tr, taille, nodata=0)
        qa = _lire(s.assets["qa_pixel"].href, crs, tr, taille, nodata=1)
        img = np.where((brut > 0) & ((qa & 0b111111) == 0), brut * 0.00341802 + 149.0 - 273.15, np.nan)
        if np.isfinite(img[disque]).mean() >= 0.3:          # au moins 30 % du disque visible
            images.append(img.astype("float32"))
    return images


def caracteristiques(lat: float, lon: float, rayon_m: int, annee: int = config.ETE_SATELLITE) -> dict:
    """lst_site (médiane 3x3 pixels au site), lst_zone (moyenne du disque), lst_ecart (médiane par date
    de site − zone), lst_rang (part du disque plus froide que le site), lst_dates. En cache."""
    cache = DOSSIER / f"lst_{lat:.4f}_{lon:.4f}_r{rayon_m}_{annee}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"  Landsat été {annee}")
    images = images_ete(lat, lon, rayon_m, annee)
    res = {"lst_site": np.nan, "lst_zone": np.nan, "lst_ecart": np.nan, "lst_rang": np.nan, "lst_dates": 0}
    if images:
        disque = _grille(lat, lon, rayon_m)[3]
        c = images[0].shape[0] // 2
        with np.errstate(all="ignore"):
            pile = np.stack(images)
            compo = np.nanmedian(pile, axis=0)
            site = float(np.nanmedian(compo[c - 1:c + 2, c - 1:c + 2]))
            zone = compo[disque]
            ecarts = [np.nanmedian(im[c - 1:c + 2, c - 1:c + 2]) - np.nanmean(im[disque]) for im in images]
            res = {"lst_site": site, "lst_zone": float(np.nanmean(zone)),
                   "lst_ecart": float(np.nanmedian(ecarts)),
                   "lst_rang": float((zone < site).sum() / np.isfinite(zone).sum()),
                   "lst_dates": len(images)}
    res = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res.items()}
    cache.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (43.3048, 5.3955)
    print(caracteristiques(lat, lon, config.RAYON_SATELLITE_M))
