"""ERA5 / ERA5-Land – réanalyse horaire au point (Copernicus CDS, jeux « time-series »).

Clé : compte gratuit sur cds.climate.copernicus.eu, puis fichier ~/.cdsapirc :
    url: https://cds.climate.copernicus.eu/api
    key: <jeton personnel>
et accepter une fois la licence (CC-BY 4.0) sur la page de chaque jeu.

Le CDS renvoie le point de grille le plus proche (0,1° pour ERA5-Land, 0,25° pour ERA5).
Horodatage UTC ; les cumuls (rayonnement, pluie) portent sur l'heure qui se termine à
l'horodatage. Si le fichier est cumulé depuis 00 UTC, il est remis en valeurs horaires.
Environ 1 min par point et par an (file d'attente du CDS).

Colonnes produites : T, TD (°C), RH (%), P (Pa), WS (m/s), WD (°), GHI (W/m², moyenne de
l'heure), IR (W/m², infrarouge descendant), RR (mm/h), TS (°C, surface) ; avec ERA5 : N (dixièmes).

Exemple :
    python api_era5.py 43.3048 5.3955 2025-01-01 2025-12-31
"""
from __future__ import annotations

import io
import sys
import zipfile

import numpy as np
import pandas as pd

import config
from outils import humidite_relative

VARIABLES = {
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m", "surface_pressure": "sp",
    "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "surface_solar_radiation_downwards": "ssrd", "surface_thermal_radiation_downwards": "strd",
    "total_precipitation": "tp", "skin_temperature": "skt",
}
JEUX = {
    "era5-land": {"id": "reanalysis-era5-land-timeseries", "grille": 0.1, "variables": VARIABLES},
    "era5": {"id": "reanalysis-era5-single-levels-timeseries", "grille": 0.25,
             "variables": {**VARIABLES, "total_cloud_cover": "tcc"}},
}
DOSSIER = config.dossier("era5")


def point_grille(jeu: str, lat: float, lon: float) -> tuple:
    g = JEUX[jeu]["grille"]
    return round(round(lat / g) * g, 4), round(round(lon / g) * g, 4)


def chemin(jeu: str, lat: float, lon: float):
    glat, glon = point_grille(jeu, lat, lon)
    return DOSSIER / f"{jeu}_{glat:.2f}_{glon:.2f}.parquet"


def _lire_csv(fichier) -> pd.DataFrame:
    """CSV du CDS (éventuellement un zip de CSV) -> tableau brut indexé en UTC."""
    if zipfile.is_zipfile(fichier):
        with zipfile.ZipFile(fichier) as z:
            parts = [pd.read_csv(io.BytesIO(z.read(n)), comment="#") for n in z.namelist() if n.endswith(".csv")]
    else:
        parts = [pd.read_csv(fichier, comment="#")]
    brut = None
    for df in parts:
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = next(c for c in ("valid_time", "time", "date") if c in df)
        df = df.rename(columns={**VARIABLES, "total_cloud_cover": "tcc"})
        garder = [c for c in df if c in set(VARIABLES.values()) | {"tcc"}]
        df.index = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None)
        df = df[garder].groupby(level=0).first()
        brut = df if brut is None else brut.join(df, how="outer", rsuffix="_bis")
    return brut[[c for c in brut if not c.endswith("_bis")]].sort_index()


def _cumule_depuis_minuit(s: pd.Series) -> bool:
    """Vrai si la série ne redescend qu'à 01 UTC (cumul journalier remis à zéro)."""
    d = s.dropna().diff().dropna()
    if len(d) < 48:
        return False
    a01 = d.index.hour == 1
    tol = 1e-6 * s.abs().max()
    return (d[~a01] < -tol).mean() < 0.01 and (d[a01] < -tol).mean() > 0.3


def _unites_physiques(brut: pd.DataFrame) -> pd.DataFrame:
    brut = brut.copy()
    if "strd" in brut and _cumule_depuis_minuit(brut["strd"]):
        for v in ("ssrd", "strd", "tp"):
            if v in brut:
                h = brut[v].diff()
                h[brut.index.hour == 1] = brut[v][brut.index.hour == 1]
                brut[v] = h.clip(lower=0)
    out = pd.DataFrame(index=brut.index)
    out["T"] = brut["t2m"] - 273.15
    out["TD"] = brut["d2m"] - 273.15
    out["RH"] = humidite_relative(out["T"], out["TD"]).clip(0, 100)
    out["P"] = brut["sp"]
    out["WS"] = np.hypot(brut["u10"], brut["v10"])
    out["WD"] = np.degrees(np.arctan2(-brut["u10"], -brut["v10"])) % 360
    out["GHI"] = (brut["ssrd"] / 3600).clip(lower=0)     # J/m² sur l'heure -> W/m² moyen
    out["IR"] = brut["strd"] / 3600
    out["RR"] = (brut["tp"] * 1000).clip(lower=0)
    out["TS"] = brut["skt"] - 273.15
    if "tcc" in brut:
        out["N"] = (brut["tcc"] * 10).clip(0, 10)
    out.index.name = "time"
    return out


def telecharger(lat: float, lon: float, debut: str, fin: str, jeu: str = config.ERA5_JEU):
    """Télécharge [debut, fin] au point de grille du site et l'ajoute au Parquet de ce point."""
    import cdsapi

    out = chemin(jeu, lat, lon)
    if out.exists():                       # déjà couvert ? on ne retélécharge pas
        idx = pd.read_parquet(out, columns=["T"]).index
        if idx.min() <= pd.Timestamp(debut) and idx.max() >= pd.Timestamp(fin) + pd.Timedelta(hours=23):
            return out
    glat, glon = point_grille(jeu, lat, lon)
    print(f"  ERA5 {jeu} ({glat}, {glon}) {debut} -> {fin}")
    brut_path = DOSSIER / f"_{jeu}_{glat:.2f}_{glon:.2f}.download"
    cdsapi.Client(quiet=True, progress=False).retrieve(
        JEUX[jeu]["id"], {"variable": list(JEUX[jeu]["variables"]),
                          "location": {"longitude": glon, "latitude": glat},
                          "date": [f"{debut}/{fin}"], "data_format": "csv"}, str(brut_path))
    nouveau = _unites_physiques(_lire_csv(brut_path))
    if out.exists():
        ancien = pd.read_parquet(out)
        nouveau = pd.concat([ancien[~ancien.index.isin(nouveau.index)], nouveau]).sort_index()
    nouveau.to_parquet(out)
    brut_path.unlink()
    return out


def lire(lat: float, lon: float, jeu: str = config.ERA5_JEU) -> pd.DataFrame:
    """Série horaire au point de grille du site (index UTC)."""
    return pd.read_parquet(chemin(jeu, lat, lon))


if __name__ == "__main__":
    lat, lon, debut, fin = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
    telecharger(lat, lon, debut, fin)
    print(lire(lat, lon).describe().round(1).to_string())
