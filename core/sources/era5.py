"""ERA5 / ERA5-Land – réanalyse horaire au point, Copernicus CDS (jeux « time-series »).

Chaîne : requête CDS pour un point et une période (le CDS prend le point de grille
le plus proche) -> fichier CSV (éventuellement zippé) -> tableau horaire en
unités physiques -> Parquet par jeu et par point de grille (les périodes
téléchargées s'ajoutent).

Clé : compte gratuit sur cds.climate.copernicus.eu, puis fichier ~/.cdsapirc :
    url: https://cds.climate.copernicus.eu/api
    key: <jeton personnel>
(ou variables d'environnement CDSAPI_URL et CDSAPI_KEY). Licence CC-BY 4.0
(usage commercial autorisé, citer Copernicus), à accepter une fois sur la page du jeu.

Conventions : horodatage UTC ; les cumuls (rayonnement, pluie) portent sur l'heure
qui se termine à l'horodatage. ERA5-Land cumule depuis 00 UTC dans ses fichiers
d'origine : le cumul journalier est détecté et converti en valeurs horaires.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core import epw
from core.config import PROCESSED_DIR, RAW_DIR

RAW = RAW_DIR / "era5"
PROC = PROCESSED_DIR / "era5"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

_COMMON = {
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m", "surface_pressure": "sp",
    "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "surface_solar_radiation_downwards": "ssrd", "surface_thermal_radiation_downwards": "strd",
    "total_precipitation": "tp", "skin_temperature": "skt",
}
DATASETS = {
    "era5-land": {
        "id": "reanalysis-era5-land-timeseries", "grille": 0.1,
        "libelle": "ERA5-Land – 0,1° (~9 km), sur terre, recommandé pour la température",
        "variables": dict(_COMMON),
    },
    "era5": {
        "id": "reanalysis-era5-single-levels-timeseries", "grille": 0.25,
        "libelle": "ERA5 – 0,25° (~28 km), avec nébulosité et rayonnement direct",
        "variables": {**_COMMON, "total_cloud_cover": "tcc",
                      "total_sky_direct_solar_radiation_at_surface": "fdir"},
    },
}
ACCUMULATED = ("ssrd", "strd", "tp", "fdir")
LATENCE_JOURS = 7   # ERA5T : ~5 jours de retard, ERA5-Land : ~6 jours

# Colonnes physiques produites (unités)
UNITES = {
    "T": "Température à 2 m (°C)", "TD": "Point de rosée à 2 m (°C)", "RH": "Humidité relative (%)",
    "P": "Pression de surface (Pa)", "WS": "Vent à 10 m (m/s)", "WD": "Direction du vent (°)",
    "GHI": "Rayonnement global (Wh/m² sur l'heure)", "IR": "Infrarouge descendant (W/m²)",
    "RR": "Pluie (mm/h)", "TS": "Température de surface (°C)", "N": "Nébulosité (dixièmes)",
    "DNI": "Rayonnement direct normal (Wh/m²)", "DHI": "Rayonnement diffus horizontal (Wh/m²)",
}


# --------------------------------------------------------------------------- #
# Clé
# --------------------------------------------------------------------------- #
def has_key() -> bool:
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return True
    rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
    if not rc.exists():
        return False
    txt = rc.read_text()
    return "url:" in txt and "key:" in txt


# --------------------------------------------------------------------------- #
# Lecture du fichier CDS
# --------------------------------------------------------------------------- #
def _read_members(path: Path) -> list[pd.DataFrame]:
    """CSV simple ou zip de CSV (un par variable, ou tout-en-un)."""
    if zipfile.is_zipfile(path):
        frames = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    frames.append(pd.read_csv(io.BytesIO(z.read(name)), comment="#"))
                elif name.lower().endswith(".nc"):
                    raise ValueError("NetCDF reçu : la requête doit demander data_format='csv'.")
        if not frames:
            raise ValueError(f"aucun CSV dans {path.name}")
        return frames
    return [pd.read_csv(path, comment="#")]


def parse(path: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    """Fichier CDS -> (tableau brut indexé en UTC, colonnes = noms courts ; point de grille)."""
    long_to_short = DATASETS[dataset]["variables"]
    shorts = set(long_to_short.values())
    merged, grid = None, {}
    for df in _read_members(path):
        cols = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=cols)
        tcol = next((c for c in ("valid_time", "time", "date", "datetime") if c in df.columns), None)
        if tcol is None:
            raise ValueError(f"colonne de temps introuvable parmi {list(df.columns)}")
        for k in ("latitude", "longitude"):
            if k in df.columns and k not in grid:
                grid[k] = float(df[k].iloc[0])
        df = df.rename(columns={k: v for k, v in long_to_short.items() if k in df.columns})
        keep = [c for c in df.columns if c in shorts]
        if not keep:
            continue
        part = df[[tcol] + keep].copy()
        part[tcol] = pd.to_datetime(part[tcol], utc=True).dt.tz_localize(None)
        part = part.groupby(tcol).first()
        merged = part if merged is None else merged.join(part, how="outer", rsuffix="_dup")
    if merged is None:
        raise ValueError("aucune variable reconnue dans le fichier CDS")
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]].sort_index()
    merged.index.name = "time"
    return merged, grid


def is_daily_accumulated(s: pd.Series) -> bool:
    """Vrai si la série est cumulée depuis 00 UTC (remise à zéro à 01 UTC)."""
    s = s.dropna()
    if len(s) < 48 or s.max() <= 0:
        return False
    d = s.diff().dropna()
    at01 = d.index.hour == 1
    tol = 1e-6 * s.abs().max()
    drops_elsewhere = (d[~at01] < -tol).mean()
    drops_at01 = (d[at01] < -tol).mean() if at01.any() else 0.0
    return drops_elsewhere < 0.01 and drops_at01 > 0.3


def deaccumulate(s: pd.Series) -> pd.Series:
    """Cumul depuis 00 UTC -> valeur de l'heure écoulée."""
    h = s.diff()
    first = s.index.hour == 1
    h[first] = s[first]
    return h.clip(lower=0)


def to_physical(raw: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Noms courts CDS -> colonnes physiques (voir UNITES)."""
    raw = raw.copy()
    # Même convention pour tous les cumuls d'un fichier : on la détecte sur l'infrarouge
    # (toujours positif, donc remis à zéro chaque jour s'il est cumulé), sinon sur le solaire.
    probe = next((raw[v] for v in ("strd", "ssrd") if v in raw), None)
    if probe is not None and is_daily_accumulated(probe):
        for v in ACCUMULATED:
            if v in raw:
                raw[v] = deaccumulate(raw[v])
    out = pd.DataFrame(index=raw.index)
    if "t2m" in raw:
        out["T"] = raw["t2m"] - 273.15
    if "d2m" in raw:
        out["TD"] = raw["d2m"] - 273.15
    if "T" in out and "TD" in out:
        out["RH"] = epw.rh_from_t_td(out["T"], out["TD"]).clip(0, 100)
    if "sp" in raw:
        out["P"] = raw["sp"]
    if "u10" in raw and "v10" in raw:
        out["WS"] = np.hypot(raw["u10"], raw["v10"])
        out["WD"] = np.degrees(np.arctan2(-raw["u10"], -raw["v10"])) % 360
    if "ssrd" in raw:
        out["GHI"] = (raw["ssrd"] / 3600).clip(lower=0)
    if "strd" in raw:
        out["IR"] = raw["strd"] / 3600
    if "tp" in raw:
        out["RR"] = (raw["tp"] * 1000).clip(lower=0)
    if "skt" in raw:
        out["TS"] = raw["skt"] - 273.15
    if "tcc" in raw:
        out["N"] = (raw["tcc"] * 10).clip(0, 10)
    if "fdir" in raw and "GHI" in out:
        direct_h = (raw["fdir"] / 3600).clip(lower=0)
        cosz = epw.solar_cos_zenith(pd.DatetimeIndex(raw.index) - pd.Timedelta(minutes=30), lat, lon)
        up = cosz > 0.087
        out["DNI"] = np.where(up, direct_h / np.where(up, cosz, 1), 0.0)
        out["DHI"] = (out["GHI"] - np.where(up, direct_h, 0.0)).clip(lower=0)
    return out


# --------------------------------------------------------------------------- #
# Téléchargement + stockage
# --------------------------------------------------------------------------- #
def grid_point(dataset: str, lat: float, lon: float) -> tuple[float, float]:
    g = DATASETS[dataset]["grille"]
    return round(round(lat / g) * g, 4), round(round(lon / g) * g, 4)


def site_path(dataset: str, lat: float, lon: float) -> Path:
    glat, glon = grid_point(dataset, lat, lon)
    return PROC / f"{dataset}_{glat:.2f}_{glon:.2f}.parquet"


def request(dataset: str, lat: float, lon: float, start: str, end: str) -> dict:
    glat, glon = grid_point(dataset, lat, lon)
    return {"variable": list(DATASETS[dataset]["variables"]),
            "location": {"longitude": glon, "latitude": glat},
            "date": [f"{start}/{end}"], "data_format": "csv"}


def fetch(dataset: str, lat: float, lon: float, start: str, end: str, client=None) -> Path:
    """Télécharge la période [start, end] (AAAA-MM-JJ) et l'ajoute au Parquet du point."""
    if client is None:
        import cdsapi
        client = cdsapi.Client(quiet=True, progress=False)
    glat, glon = grid_point(dataset, lat, lon)
    raw_path = RAW / f"{dataset}_{glat:.2f}_{glon:.2f}_{start}_{end}.download"
    client.retrieve(DATASETS[dataset]["id"], request(dataset, lat, lon, start, end), str(raw_path))
    raw, grid = parse(raw_path, dataset)
    phys = to_physical(raw, grid.get("latitude", glat), grid.get("longitude", glon))
    out = site_path(dataset, lat, lon)
    if out.exists():
        old = pd.read_parquet(out)
        phys = pd.concat([old[~old.index.isin(phys.index)], phys]).sort_index()
    phys.attrs = {}
    phys.to_parquet(out)
    meta = out.with_suffix(".json")
    meta.write_text(pd.Series({"dataset": dataset, "lat_grille": grid.get("latitude", glat),
                               "lon_grille": grid.get("longitude", glon)}).to_json())
    raw_path.unlink(missing_ok=True)
    return out


def is_ready(dataset: str, lat: float, lon: float) -> bool:
    return site_path(dataset, lat, lon).exists()


def load(dataset: str, lat: float, lon: float) -> pd.DataFrame:
    return pd.read_parquet(site_path(dataset, lat, lon))


def grid_info(dataset: str, lat: float, lon: float) -> dict:
    meta = site_path(dataset, lat, lon).with_suffix(".json")
    return pd.read_json(meta, typ="series").to_dict() if meta.exists() else {}
