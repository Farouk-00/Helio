"""Cerema – zones climatiques locales (LCZ), 93 territoires urbains (2022), sans clé.

- Liste des territoires : service ArcGIS public du Cerema (contour + lien de téléchargement).
- Données : un zip par territoire sur data.gouv (20 à 230 Mo : shapefile Lambert-93 + raster).
  Seul le shapefile est gardé (converti en GeoParquet, quelques Mo), le zip est supprimé.

Classes : 1 à 9 (bâti, de « compact de tours » à « maisons diffuses »), A à G (arbres,
végétation basse, sol imperméable, sol nu, eau). Pas de LCZ 10 en France.
Indicateurs par polygone (sens d'après le guide Cerema, à confirmer) : hre hauteur du bâti,
are rapport H/L des rues, bur part bâtie, ror part imperméable, bsr sol nu, war eau,
ver végétation, vhr végétation haute.

Exemple :
    python api_cerema_lcz.py 43.3048 5.3955
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import config
from outils import get, telecharger_fichier

TERRITOIRES_URL = ("https://cartagene.cerema.fr/server/rest/services/Hosted/"
                   "aires_urbaines_3857/FeatureServer/0/query")
RESSOURCE_DATAGOUV = "https://www.data.gouv.fr/fr/datasets/r/{rid}"
CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G"]
INDICATEURS = ["hre", "are", "bur", "ror", "bsr", "war", "ver", "vhr"]
DOSSIER = config.dossier("lcz")
_CODE = re.compile(r"^(?:LCZ)?[\s_:-]*(?:(\d{1,3})(?:\.0+)?(?!\d)|([A-G])(?![A-Z0-9]))")


def code_lcz(valeur) -> str | None:
    """'2', 2.0, 'LCZ 2', 'a', 11 ou 101 (codage WUDAPT) -> '2' / 'A'."""
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return None
    m = _CODE.match(str(valeur).strip().upper())
    if not m:
        return None
    if m.group(2):
        return m.group(2)
    n = int(m.group(1))
    if 1 <= n <= 10:
        return str(n)
    if 11 <= n <= 17:
        return "ABCDEFG"[n - 11]
    if 101 <= n <= 107:
        return "ABCDEFG"[n - 101]
    return None


def territoires() -> gpd.GeoDataFrame:
    """Les 93 territoires : code, nom, url du zip, contour (WGS84). Mis en cache."""
    cache = DOSSIER / "_territoires.geojson"
    if not cache.exists():
        r = get(TERRITOIRES_URL, params={"where": "1=1", "outFields": "fua_code,fua_name,ddl_link",
                                         "returnGeometry": "true", "outSR": 4326,
                                         "maxAllowableOffset": 0.0005, "f": "geojson"})
        g = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        rid = g["ddl_link"].str.extract(r"/resources/([0-9a-f-]{36})", expand=False)
        g = gpd.GeoDataFrame({"code": g["fua_code"], "nom": g["fua_name"],
                              "url": [RESSOURCE_DATAGOUV.format(rid=x) if isinstance(x, str) else None for x in rid]},
                             geometry=g.geometry, crs="EPSG:4326")
        g.to_file(cache, driver="GeoJSON")
    return gpd.read_file(cache)


def territoire_du_point(lat: float, lon: float) -> str | None:
    t = territoires()
    hit = t[t.contains(Point(lon, lat))]
    return None if hit.empty else hit.iloc[0]["code"]


def chemin(code: str) -> Path:
    return DOSSIER / f"lcz_{code}.parquet"


def telecharger(code: str) -> Path:
    """Zip du territoire -> GeoParquet (colonne lcz_code + indicateurs), zip supprimé."""
    out = chemin(code)
    if out.exists():
        return out
    t = territoires()
    url = t.loc[t["code"] == code, "url"].iloc[0]
    zip_path = DOSSIER / f"lcz_{code}.zip"
    print(f"  LCZ territoire {code} (zip de 20 à 230 Mo)")
    telecharger_fichier(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:     # le plus gros shapefile, de préférence nommé *lcz*
        shps = [i for i in z.infolist() if i.filename.lower().endswith(".shp")]
    shps.sort(key=lambda i: ("lcz" in Path(i.filename).name.lower(), i.file_size), reverse=True)
    g = gpd.read_file(f"/vsizip/{zip_path.resolve()}/{shps[0].filename}")
    g.columns = [c.lower() if c != "geometry" else c for c in g.columns]
    g = g.set_geometry("geometry")
    colonne = next(c for c in [c for c in g if c.startswith("lcz")] + list(g.columns)
                   if c != "geometry" and g[c].head(500).map(code_lcz).notna().mean() > 0.8)
    g["lcz_code"] = g[colonne].map(code_lcz)
    if not g.crs.is_projected:
        g = g.to_crs(g.estimate_utm_crs())
    g[["lcz_code"] + [c for c in INDICATEURS if c in g] + ["geometry"]].to_parquet(out, index=False)
    zip_path.unlink()
    return out


def caracteristiques(lat: float, lon: float, rayon_m: float) -> dict:
    """Classe LCZ au site (lcz_site_X = 1), part de chaque classe et indicateurs moyens dans le rayon.

    Tout est NaN si le site est hors des 93 territoires.
    """
    res = {f"lcz_site_{c}": np.nan for c in CLASSES}
    res.update({f"lcz_part_{c}": np.nan for c in CLASSES})
    res.update({f"lcz_{c}": np.nan for c in INDICATEURS})
    code = territoire_du_point(lat, lon)
    if code is None:
        return res
    g = gpd.read_parquet(telecharger(code))
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(g.crs).iloc[0]
    disque = site.buffer(rayon_m)
    proche = g.iloc[g.sindex.query(disque, predicate="intersects")].copy()
    if proche.empty:
        return res
    proche["geometry"] = proche.geometry.make_valid().intersection(disque)
    surface = proche.geometry.area
    parts = surface.groupby(proche["lcz_code"]).sum() / surface.sum()
    au_site = g.iloc[g.sindex.query(site, predicate="intersects")]["lcz_code"]
    for c in CLASSES:
        res[f"lcz_part_{c}"] = float(parts.get(c, 0.0))
        res[f"lcz_site_{c}"] = float(len(au_site) > 0 and au_site.iloc[0] == c)
    for c in INDICATEURS:
        if c in proche and proche[c].notna().any():
            v = pd.to_numeric(proche[c], errors="coerce")
            res[f"lcz_{c}"] = float((v * surface).sum() / surface[v.notna()].sum())
    return res


if __name__ == "__main__":
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (43.3048, 5.3955)
    print(territoire_du_point(lat, lon))
    print({k: v for k, v in caracteristiques(lat, lon, config.RAYON_URBAIN_M).items() if v == v and v != 0})
