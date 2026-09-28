"""LCZ Cerema – « Cartographie des zones climatiques locales » (93 territoires, 2022).

Chaîne : liste des territoires (service ArcGIS public du Cerema : contour + lien
data.gouv) -> téléchargement du zip d'un territoire (shapefile + raster 1,5 m,
100-230 Mo) -> conversion du shapefile seul en GeoParquet -> lecture par l'appli.

Les territoires sont des aires urbaines, pas des départements : on rattache un
territoire à un département dès qu'il contient le centre d'une de ses communes.
Les géométries sont gardées dans leur projection d'origine (en mètres :
Lambert-93 en métropole, UTM en outre-mer) pour les calculs de surface.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from core.config import (DATAGOUV_RESOURCE, LCZ_COMMUNES_URL, LCZ_TERRITOIRES_URL,
                         PROCESSED_DIR, RAW_DIR)
from core.net import download, get_json

RAW = RAW_DIR / "lcz"
PROC = PROCESSED_DIR / "lcz"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
CATALOG_CACHE = RAW / "_territoires.geojson"

# Libellés et couleurs de la légende officielle Cerema (pas de LCZ 10 en France)
LCZ_CLASSES = {
    "1": ("Ensemble compact de tours", "#8c0000"),
    "2": ("Ensemble compact d'immeubles", "#d10000"),
    "3": ("Ensemble compact de maisons", "#ff0000"),
    "4": ("Ensemble de tours espacées", "#bf4d00"),
    "5": ("Ensemble d'immeubles espacés", "#ff6600"),
    "6": ("Ensemble de maisons espacées", "#ff9955"),
    "7": ("Ensemble dense de constructions légères", "#faee05"),
    "8": ("Bâtiments de grande emprise", "#bcbcbc"),
    "9": ("Implantation diffuse de maisons", "#ffccaa"),
    "10": ("Industrie lourde", "#555555"),
    "A": ("Espace densément arboré", "#006a00"),
    "B": ("Espace arboré clairsemé", "#00aa00"),
    "C": ("Espace végétalisé hétérogène", "#648525"),
    "D": ("Végétation basse", "#b9db79"),
    "E": ("Sol imperméable naturel ou artificiel", "#000000"),
    "F": ("Sol nu perméable", "#fbf7ae"),
    "G": ("Surface en eau", "#6a6aff"),
}

# Indicateurs par polygone (sens supposé d'après le guide Cerema – à confirmer)
INDICATEURS = {
    "hre": "Hauteur moyenne du bâti (m)",
    "are": "Rapport d'aspect H/L des rues",
    "bur": "Part de surface bâtie",
    "ror": "Part de surface imperméable (hors bâti)",
    "bsr": "Part de sol nu",
    "war": "Part d'eau",
    "ver": "Part de végétation",
    "vhr": "Part de végétation haute",
}


_CODE_RE = re.compile(r"^(?:LCZ)?[\s_:-]*(?:(\d{1,3})(?:\.0+)?(?!\d)|([A-G])(?![A-Z0-9]))")


def label(code: str | None) -> str:
    if code not in LCZ_CLASSES:
        return "hors LCZ"
    return f"LCZ {code} – {LCZ_CLASSES[code][0]}"


def normalize_code(value) -> str | None:
    """'2', 2, 2.0, 'LCZ 2', 'LCZ 2 : Ensemble…', 'a', 'lcz_a', 11 ou 101 (WUDAPT) -> '2' / 'A'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = _CODE_RE.match(str(value).strip().upper())
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


# --------------------------------------------------------------------------- #
# Catalogue des territoires
# --------------------------------------------------------------------------- #
def get_catalog(refresh: bool = False) -> gpd.GeoDataFrame:
    """Territoires LCZ : code, nom, url du zip, contour (WGS84)."""
    if CATALOG_CACHE.exists() and not refresh:
        gdf = gpd.read_file(CATALOG_CACHE)
    else:
        r = get_json(LCZ_TERRITOIRES_URL, params={
            "where": "1=1", "outFields": "fua_code,fua_name,ddl_link",
            "returnGeometry": "true", "outSR": 4326, "maxAllowableOffset": 0.0005,
            "f": "geojson",
        })
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        rid = gdf["ddl_link"].str.extract(r"/resources/([0-9a-f-]{36})", expand=False)
        gdf = gpd.GeoDataFrame({
            "code": gdf["fua_code"], "nom": gdf["fua_name"],
            "url": rid.map(lambda r: DATAGOUV_RESOURCE.format(rid=r) if isinstance(r, str) else None),
        }, geometry=gdf.geometry, crs="EPSG:4326")
        gdf.to_file(CATALOG_CACHE, driver="GeoJSON")
    return gdf.sort_values("nom").reset_index(drop=True)


def _communes_centres(departement: str) -> gpd.GeoDataFrame:
    """Un point par commune du département (Corse « 20 » = 2A + 2B)."""
    cache = RAW / f"_communes_{departement}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        prefixes = ["2A", "2B"] if departement == "20" else [departement]
        where = " OR ".join(f"insee_com LIKE '{p}%'" for p in prefixes)
        r = get_json(LCZ_COMMUNES_URL, params={
            "where": where, "outFields": "insee_com,nom", "returnGeometry": "true",
            "outSR": 4326, "maxAllowableOffset": 0.002, "resultRecordCount": 2000,
            "f": "geojson",
        })
        feats = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        pts = feats.geometry.representative_point() if not feats.empty else []
        rows = [{"code": c, "nom": n, "lon": p.x, "lat": p.y}
                for c, n, p in zip(feats.get("insee_com", []), feats.get("nom", []), pts)]
        cache.write_text(json.dumps(rows))
    df = pd.DataFrame(rows, columns=["code", "nom", "lon", "lat"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def territories_for(departement: str, catalog: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Territoires qui contiennent au moins un centre de commune du département."""
    communes = _communes_centres(departement)
    if communes.empty or catalog.empty:
        return catalog.iloc[0:0].assign(n_communes=0)
    terr = catalog[["code", "geometry"]].rename(columns={"code": "territoire"})
    hits = gpd.sjoin(communes, terr, predicate="within")
    n = hits.groupby("territoire").size().rename("n_communes")
    return catalog.merge(n, left_on="code", right_index=True)


def territory_at(lat: float, lon: float, catalog: gpd.GeoDataFrame) -> str | None:
    """Code du territoire qui contient le point, s'il y en a un."""
    hit = catalog[catalog.contains(Point(lon, lat))]
    return None if hit.empty else hit.iloc[0]["code"]


# --------------------------------------------------------------------------- #
# Téléchargement + conversion
# --------------------------------------------------------------------------- #
def parquet_path(code: str) -> Path:
    return PROC / f"lcz_{code}.parquet"


def downloaded() -> list[str]:
    return sorted(p.stem[len("lcz_"):] for p in PROC.glob("lcz_*.parquet"))


def _pick_shapefile(zip_path: Path) -> str:
    """Le shapefile LCZ du zip (le plus gros .shp, de préférence nommé *lcz*)."""
    with zipfile.ZipFile(zip_path) as z:
        shps = [i for i in z.infolist() if i.filename.lower().endswith(".shp")]
    if not shps:
        raise ValueError(f"aucun shapefile dans {zip_path.name}")
    shps.sort(key=lambda i: ("lcz" in Path(i.filename).name.lower(), i.file_size), reverse=True)
    return shps[0].filename


def _code_column(gdf: gpd.GeoDataFrame) -> str:
    """Colonne qui porte la classe LCZ (la première dont les valeurs se décodent)."""
    candidates = [c for c in gdf.columns if c.startswith("lcz")] + \
                 [c for c in gdf.columns if not c.startswith("lcz") and c != "geometry"]
    for col in candidates:
        ok = gdf[col].head(500).map(normalize_code).notna().mean()
        if ok > 0.8:
            return col
    raise ValueError(f"colonne de classe LCZ introuvable parmi {list(gdf.columns)}")


def shapefile_to_parquet(zip_path: Path, dest: Path) -> gpd.GeoDataFrame:
    """Lit le shapefile LCZ directement dans le zip et l'écrit en GeoParquet."""
    member = _pick_shapefile(zip_path)
    gdf = gpd.read_file(f"/vsizip/{zip_path.resolve()}/{member}")
    gdf.columns = [c.lower() if c != "geometry" else c for c in gdf.columns]
    gdf = gdf.set_geometry("geometry")
    gdf["lcz_code"] = gdf[_code_column(gdf)].map(normalize_code)
    if gdf.crs is not None and not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf.to_parquet(dest, index=False)
    return gdf


def fetch(code: str, url: str, progress=None, keep_raw: bool = False) -> Path:
    """Télécharge et convertit un territoire."""
    raw = RAW / f"lcz_{code}.zip"
    if not raw.exists():
        download(url, raw, progress)
    out = parquet_path(code)
    shapefile_to_parquet(raw, out)
    if not keep_raw:
        raw.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def load(code: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(parquet_path(code))


def around(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_m: float) -> gpd.GeoDataFrame:
    """Polygones découpés au disque de rayon radius_m autour du point, avec leur surface."""
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
    disk = site.buffer(radius_m)
    near = gdf.iloc[gdf.sindex.query(disk, predicate="intersects")].copy()
    if near.empty:
        return near.assign(surface_m2=[])
    near["geometry"] = near.geometry.make_valid().intersection(disk)
    near = near[~near.geometry.is_empty]
    near["surface_m2"] = near.geometry.area
    return near


def site_profile(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_m: float) -> dict:
    """Classe LCZ au point, composition et indicateurs moyens dans le rayon."""
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
    at = gdf.iloc[gdf.sindex.query(site, predicate="intersects")]
    near = around(gdf, lat, lon, radius_m)
    if near.empty:
        return {"classe": None, "polygone": None, "composition": pd.DataFrame(),
                "indicateurs": pd.Series(dtype=float), "near": near}
    comp = (near.groupby("lcz_code")["surface_m2"].sum() / near["surface_m2"].sum()).rename("part")
    comp = comp.sort_values(ascending=False).reset_index()
    comp["libellé"] = comp["lcz_code"].map(label)
    w = near["surface_m2"]
    ind = {INDICATEURS.get(c, c): (near[c] * w).sum() / w[near[c].notna()].sum()
           for c in INDICATEURS if c in near and pd.api.types.is_numeric_dtype(near[c])
           and near[c].notna().any()}
    return {
        "classe": at.iloc[0]["lcz_code"] if not at.empty else None,
        "polygone": at.iloc[0].drop("geometry") if not at.empty else None,
        "composition": comp,
        "indicateurs": pd.Series(ind, dtype=float),
        "near": near,
    }
