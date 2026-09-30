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


def _elevations(lats: list[float], lons: list[float]) -> np.ndarray:
    """Altitudes de plusieurs points en une requête (NaN hors couverture, ex. en mer)."""
    r = get_json(ALTI_URL, params={"lon": "|".join(f"{x:.6f}" for x in lons),
                                   "lat": "|".join(f"{y:.6f}" for y in lats),
                                   "resource": "ign_rge_alti_wld", "zonly": "true", "delimiter": "|"})
    z = np.array(r.json().get("elevations", []), dtype=float)
    if len(z) != len(lats):
        raise ValueError(f"altimétrie : {len(z)} valeurs reçues pour {len(lats)} points")
    return np.where(z <= -9999, np.nan, z)


def relief(lat: float, lon: float, grid_deg: float = 0.1) -> dict:
    """Position du site dans le relief (utile la nuit : l'air froid s'accumule dans les creux).

    - tpi_500, tpi_2000 : altitude du site moins l'altitude moyenne d'un cercle de 500 m / 2 km
      (négatif = creux, positif = hauteur) ;
    - alt_maille : altitude moyenne de la maille ERA5 qui contient le site (5 x 5 points) ;
    - ecart_alt_maille : altitude du site moins celle de la maille (ERA5 ignore cet écart).
    """
    cache = _cache()
    k = f"relief:{grid_deg}:" + _key(lat, lon)
    if k not in cache:
        lats, lons = [lat], [lon]
        m_lat = 1 / 111_320
        m_lon = 1 / (111_320 * np.cos(np.radians(lat)))
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for radius in (500, 2000):
            lats += list(lat + radius * m_lat * np.sin(angles))
            lons += list(lon + radius * m_lon * np.cos(angles))
        glat, glon = round(lat / grid_deg) * grid_deg, round(lon / grid_deg) * grid_deg
        offs = np.linspace(-grid_deg / 2 * 0.8, grid_deg / 2 * 0.8, 5)
        for dy in offs:
            for dx in offs:
                lats.append(glat + dy)
                lons.append(glon + dx)
        z = _elevations(lats, lons)
        z0, ring500, ring2000, cell = z[0], z[1:13], z[13:25], z[25:]
        with np.errstate(all="ignore"):
            res = {"tpi_500": z0 - np.nanmean(ring500), "tpi_2000": z0 - np.nanmean(ring2000),
                   "alt_maille": np.nanmean(cell)}
        res["ecart_alt_maille"] = z0 - res["alt_maille"]
        cache[k] = {n: (None if not np.isfinite(v) else round(float(v), 1)) for n, v in res.items()}
        CACHE.write_text(json.dumps(cache))
    return {n: (np.nan if v is None else v) for n, v in cache[k].items()}


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
