"""Stations Météo-France – « Données climatologiques de base – horaires ».

Chaîne : catalogue data.gouv -> téléchargement d'un fichier CSV.gz
(1 département x 1 période) -> conversion en Parquet (colonnes utiles)
-> lecture par l'appli.

Format source : séparateur « ; », une ligne par station et par heure.
Heures en UTC pour la métropole, en heure locale (« FU ») pour l'outre-mer.
Chaque mesure X a une colonne qualité QX (0 protégée, 1 validée,
2 douteuse, 9 filtrée non encore validée).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from core.config import (DATAGOUV_API, DATASET_HORAIRE_ID, HTTP_TIMEOUT,
                         PROCESSED_DIR, RAW_DIR)

RAW = RAW_DIR / "stations_horaires"
PROC = PROCESSED_DIR / "stations_horaires"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
CATALOG_CACHE = RAW / "_catalogue.json"

FILE_RE = re.compile(r"H_(\d{2,3})_(.+?)\.csv(\.gz)?$", re.IGNORECASE)

# Colonnes conservées (les autres sont ignorées pour limiter la taille)
ID_COLS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH"]
MEASURES = {
    "T": "Température sous abri (°C)",
    "TD": "Point de rosée (°C)",
    "TN": "Température minimale de l'heure (°C)",
    "TX": "Température maximale de l'heure (°C)",
    "U": "Humidité relative (%)",
    "FF": "Vent moyen 10 min à 10 m (m/s)",
    "DD": "Direction du vent (°)",
    "FXI": "Rafale maximale (m/s)",
    "RR1": "Pluie sur l'heure (mm)",
    "GLO": "Rayonnement global (J/cm²)",
    "INS": "Durée d'ensoleillement (min)",
    "N": "Nébulosité totale (octas)",
    "PMER": "Pression au niveau de la mer (hPa)",
}
KEEP = set(ID_COLS) | set(MEASURES) | {f"Q{m}" for m in MEASURES}


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def _fetch_resources() -> list[dict]:
    """Liste des fichiers du jeu de données (API v2 paginée, repli sur v1)."""
    resources: list[dict] = []
    url = f"{DATAGOUV_API}/2/datasets/{DATASET_HORAIRE_ID}/resources/?page_size=200"
    try:
        while url:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            resources += payload.get("data", [])
            url = payload.get("next_page")
        if resources:
            return resources
    except requests.RequestException:
        pass
    r = requests.get(f"{DATAGOUV_API}/1/datasets/{DATASET_HORAIRE_ID}/", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("resources", [])


def get_catalog(refresh: bool = False) -> pd.DataFrame:
    """Tableau (departement, periode, fichier, url, taille_mo) des fichiers horaires."""
    if CATALOG_CACHE.exists() and not refresh:
        rows = json.loads(CATALOG_CACHE.read_text())
    else:
        rows = []
        for res in _fetch_resources():
            url = res.get("url") or res.get("latest") or ""
            name = url.rsplit("/", 1)[-1] or res.get("title", "")
            m = FILE_RE.search(name) or FILE_RE.search(res.get("title", ""))
            if not m:
                continue
            size = res.get("filesize") or 0
            rows.append({
                "departement": m.group(1),
                "periode": m.group(2),
                "fichier": m.group(0),
                "url": url,
                "taille_mo": round(size / 1e6, 1) if size else None,
            })
        CATALOG_CACHE.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    df = pd.DataFrame(rows, columns=["departement", "periode", "fichier", "url", "taille_mo"])
    return df.sort_values(["departement", "periode"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Téléchargement + conversion
# --------------------------------------------------------------------------- #
def parquet_path(departement: str, periode: str) -> Path:
    return PROC / f"H_{departement}_{periode}.parquet"


def is_ready(departement: str, periode: str) -> bool:
    return parquet_path(departement, periode).exists()


def download(url: str, dest: Path, progress=None) -> Path:
    """Téléchargement en flux ; progress(fraction) est optionnel."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(done / total, 1.0))
    tmp.rename(dest)
    return dest


def csv_to_parquet(src: Path, dest: Path) -> pd.DataFrame:
    """Lit le CSV Météo-France (colonnes utiles seulement) et l'écrit en Parquet."""
    df = pd.read_csv(
        src, sep=";", usecols=lambda c: c.strip() in KEEP,
        dtype={"NUM_POSTE": str, "NOM_USUEL": str}, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col not in ("NUM_POSTE", "NOM_USUEL") and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    df["time"] = pd.to_datetime(df["AAAAMMJJHH"].astype("Int64").astype(str),
                                format="%Y%m%d%H", errors="coerce")
    df = df.drop(columns=["AAAAMMJJHH"]).dropna(subset=["time"])
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    df.to_parquet(dest, index=False)
    return df


def fetch(departement: str, periode: str, url: str, progress=None,
          keep_raw: bool = False) -> Path:
    """Télécharge et convertit un fichier (département x période)."""
    raw = RAW / f"H_{departement}_{periode}.csv.gz"
    if not raw.exists():
        download(url, raw, progress)
    out = parquet_path(departement, periode)
    csv_to_parquet(raw, out)
    if not keep_raw:
        raw.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def load(departement: str, periodes: list[str] | None = None) -> pd.DataFrame:
    """Toutes les observations téléchargées pour un département."""
    files = sorted(PROC.glob(f"H_{departement}_*.parquet"))
    if periodes:
        files = [f for f in files if f.stem.split("_", 2)[2] in periodes]
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.drop_duplicates(["NUM_POSTE", "time"]).sort_values(["NUM_POSTE", "time"])


def downloaded_periods(departement: str) -> list[str]:
    return sorted(f.stem.split("_", 2)[2] for f in PROC.glob(f"H_{departement}_*.parquet"))


def stations_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par station : position, période couverte, part d'heures avec T."""
    if df.empty:
        return df
    g = df.groupby("NUM_POSTE")
    out = g.agg(nom=("NOM_USUEL", "first"), lat=("LAT", "first"), lon=("LON", "first"),
                alti=("ALTI", "first"), debut=("time", "min"), fin=("time", "max"),
                n_heures=("time", "size"))
    for m in ("T", "U", "GLO"):
        if m in df.columns:
            out[f"part_{m}"] = g[m].apply(lambda s: s.notna().mean()).round(2)
    return out.reset_index()


def filter_quality(df: pd.DataFrame, var: str, codes=(0, 1)) -> pd.Series:
    """Série `var` en ne gardant que les valeurs de qualité acceptée."""
    s = df[var]
    q = f"Q{var}"
    if q in df.columns:
        s = s.where(df[q].isin(codes))
    return s
