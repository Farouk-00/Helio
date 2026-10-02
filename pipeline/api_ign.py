"""IGN Géoplateforme – altitude, position dans le relief et distance à la mer (sans clé).

- Altitude : service d'altimétrie, ressource RGE ALTI (plusieurs points par requête,
  séparés par « | »). Renvoie -99999 hors couverture (en mer).
- Distance à la mer : WFS, couche BD CARTO « limite_terre_mer » (inclut les lagunes
  ouvertes, ex. étang de Berre). Attention : la BBOX doit préciser le système
  « urn:ogc:def:crs:OGC:1.3:CRS84 » (ordre lon, lat), sinon le service renvoie 0 objet.
  Au-delà de ~50 km de la côte : 999 km (la variable n'a plus d'effet à cette distance).

Les résultats sont mis en cache par point dans donnees/ign/points.json.

Exemple :
    python api_ign.py 43.3048 5.3955
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyproj
from shapely.geometry import Point, shape
from shapely.ops import transform

import config
from outils import get

ALTI_URL = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
WFS_URL = "https://data.geopf.fr/wfs/ows"
CACHE = config.dossier("ign") / "points.json"


def _cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def altitudes(lats: list, lons: list) -> np.ndarray:
    """Altitude (m) de plusieurs points en une requête ; NaN en mer."""
    r = get(ALTI_URL, params={"lon": "|".join(f"{x:.6f}" for x in lons),
                              "lat": "|".join(f"{y:.6f}" for y in lats),
                              "resource": "ign_rge_alti_wld", "zonly": "true", "delimiter": "|"})
    z = np.array(r.json()["elevations"], dtype=float)
    return np.where(z <= -9999, np.nan, z)


def relief(lat: float, lon: float, grille_era5: float = 0.1) -> dict:
    """Altitude du site et sa position dans le relief.

    - altitude_m ;
    - tpi_500, tpi_2000 : altitude du site − altitude moyenne d'un cercle de 500 m / 2 km
      (négatif = creux où l'air froid s'accumule la nuit, positif = hauteur) ;
    - alt_maille : altitude moyenne de la maille ERA5 du site (5 x 5 points) ;
    - ecart_alt_maille : altitude du site − alt_maille (ERA5 ignore cet écart).
    """
    lats, lons = [lat], [lon]
    m_lat, m_lon = 1 / 111_320, 1 / (111_320 * np.cos(np.radians(lat)))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for rayon in (500, 2000):
        lats += list(lat + rayon * m_lat * np.sin(angles))
        lons += list(lon + rayon * m_lon * np.cos(angles))
    glat, glon = round(lat / grille_era5) * grille_era5, round(lon / grille_era5) * grille_era5
    pas = np.linspace(-0.4 * grille_era5, 0.4 * grille_era5, 5)
    for dy in pas:
        for dx in pas:
            lats.append(glat + dy)
            lons.append(glon + dx)
    z = altitudes(lats, lons)
    with np.errstate(all="ignore"):
        alt_maille = np.nanmean(z[25:])
        return {"altitude_m": z[0], "tpi_500": z[0] - np.nanmean(z[1:13]),
                "tpi_2000": z[0] - np.nanmean(z[13:25]), "alt_maille": alt_maille,
                "ecart_alt_maille": z[0] - alt_maille}


def distance_mer_km(lat: float, lon: float) -> float:
    """Distance au trait de côte IGN le plus proche (km) ; 999 au-delà de ~50 km."""
    vers_metres = pyproj.Transformer.from_crs(
        "EPSG:4326", f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m", always_xy=True).transform
    for d in (0.05, 0.2, 0.6):                       # fenêtres de plus en plus grandes (degrés)
        r = get(WFS_URL, params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": "BDCARTO_V5:limite_terre_mer", "OUTPUTFORMAT": "application/json",
            "SRSNAME": "EPSG:4326", "COUNT": 5000,
            "BBOX": f"{lon - d},{lat - d},{lon + d},{lat + d},urn:ogc:def:crs:OGC:1.3:CRS84"})
        lignes = [shape(f["geometry"]) for f in r.json().get("features", []) if f.get("geometry")]
        if lignes:
            km = min(transform(vers_metres, g).distance(Point(0, 0)) for g in lignes) / 1000
            if km <= d * 111 * np.cos(np.radians(abs(lat) + d)):   # plus proche que le bord de la fenêtre
                return km
    return 999.0


def caracteristiques(lat: float, lon: float, grille_era5: float = 0.1) -> dict:
    """Variables fixes du site issues de l'IGN (en cache)."""
    cache = _cache()
    cle = f"{lat:.5f},{lon:.5f},{grille_era5}"
    if cle not in cache:
        res = relief(lat, lon, grille_era5)
        res["dist_mer_km"] = distance_mer_km(lat, lon)
        cache[cle] = {k: (None if not np.isfinite(v) else round(float(v), 2)) for k, v in res.items()}
        CACHE.write_text(json.dumps(cache, indent=1))
    return {k: (np.nan if v is None else v) for k, v in cache[cle].items()}


if __name__ == "__main__":
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (43.3048, 5.3955)
    print(caracteristiques(lat, lon))
