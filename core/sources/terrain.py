"""Altitude (IGN RGE ALTI) et distance à la mer (IGN BD CARTO, repli Natural Earth), via la Géoplateforme."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from core.config import ALTI_URL, COASTLINE_URL, IGN_WFS_URL, PROCESSED_DIR, RAW_DIR
from core.net import download, get_json

RAW = RAW_DIR / "terrain"
PROC = PROCESSED_DIR / "terrain"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
COAST = PROC / "ne_10m_coastline.parquet"
CACHE = PROC / "_points.json"


def _cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def altitude(lat: float, lon: float) -> float | None:
    """Altitude du sol (m) ; None si le service ne couvre pas le point."""
    cache = _cache()
    k = "alt:" + _key(lat, lon)
    if k not in cache:
        r = get_json(ALTI_URL, params={"lon": lon, "lat": lat, "resource": "ign_rge_alti_wld", "zonly": "true"})
        z = r.json().get("elevations", [None])[0]
        cache[k] = None if z is None or z <= -99999 else float(z)
        CACHE.write_text(json.dumps(cache))
    return cache[k]


def coastline() -> gpd.GeoDataFrame:
    if not COAST.exists():
        raw = RAW / "ne_10m_coastline.geojson"
        if not raw.exists():
            download(COASTLINE_URL, raw)
        gpd.read_file(raw)[["geometry"]].to_parquet(COAST)
    return gpd.read_parquet(COAST)


def _ign_coast(lon: float, lat: float, d: float) -> gpd.GeoDataFrame:
    """Segments de la limite terre-mer (IGN BD CARTO) dans une fenêtre de ± d degrés."""
    r = get_json(IGN_WFS_URL, params={
        "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
        "TYPENAMES": "BDCARTO_V5:limite_terre_mer", "OUTPUTFORMAT": "application/json",
        "SRSNAME": "EPSG:4326", "COUNT": 5000,
        # l'ordre des axes n'est sans ambiguïté qu'avec le système explicite (lon, lat en CRS84)
        "BBOX": f"{lon - d},{lat - d},{lon + d},{lat + d},urn:ogc:def:crs:OGC:1.3:CRS84"})
    feats = r.json().get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")[["geometry"]]


def _nearest_km(lines: gpd.GeoDataFrame, lat: float, lon: float) -> float:
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    utm = pt.estimate_utm_crs()
    return float(lines.to_crs(utm).distance(pt.to_crs(utm).iloc[0]).min() / 1000)


def distance_mer_km(lat: float, lon: float) -> float:
    """Distance au trait de côte le plus proche (km).

    Jusqu'à ~50 km : limite terre-mer IGN BD CARTO (précision de l'ordre de 10 m).
    Au-delà : trait de côte Natural Earth 1:10 M (écart de l'ordre du km, sans effet à cette
    distance), aussi utilisé si le service IGN ne répond pas.
    """
    cache = _cache()
    k = "mer:" + _key(lat, lon)
    if k not in cache:
        dist, source = np.nan, None
        try:
            for d in (0.05, 0.2, 0.6):
                lines = _ign_coast(lon, lat, d)
                if not lines.empty:
                    km = _nearest_km(lines, lat, lon)
                    half_km = d * 111 * np.cos(np.radians(abs(lat) + d))  # demi-largeur la plus courte
                    if km <= half_km:  # plus proche que le bord de la fenêtre : c'est le vrai minimum
                        dist, source = km, "IGN BD CARTO"
                        break
        except Exception:
            pass
        if source is None:
            coast = coastline()
            for d in (1.5, 5.0, 15.0):
                near = coast.iloc[coast.sindex.query(box(lon - d, lat - d, lon + d, lat + d))]
                if not near.empty:
                    km = _nearest_km(near, lat, lon)
                    if km <= d * 111 * np.cos(np.radians(abs(lat) + d)):
                        dist, source = km, "Natural Earth"
                        break
        cache[k] = round(dist, 3)
        cache["src:" + _key(lat, lon)] = source
        CACHE.write_text(json.dumps(cache))
    return cache[k]
