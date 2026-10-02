"""Météo-France – mesures horaires des stations (« Données climatologiques de base – horaires »).

Source : data.gouv.fr, un fichier CSV.gz par département et par période, sans clé.
Les périodes sont par exemple « 2020-2023 », « previous-2024 », « latest-2025-2026 »
(le fichier « latest » est mis à jour régulièrement et contient les mois récents).

Format : séparateur « ; », une ligne par station et par heure, colonne AAAAMMJJHH.
Heures en UTC en métropole, en heure locale normale en outre-mer.
Chaque mesure X a une colonne de qualité QX : 0 protégée, 1 validée, 2 douteuse, 9 non validée.
T est la température sous abri (°C), mesurée à l'heure pile.

Exemple :
    python api_meteofrance_stations.py 84
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

import config
from outils import get, telecharger_fichier

JEU_ID = "6569b4473bedf2e7abad3b72"
API = "https://www.data.gouv.fr/api"
NOM_FICHIER = re.compile(r"H_(\d{2,3})_(.+?)\.csv(\.gz)?$", re.IGNORECASE)
COLONNES = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJHH",
            "T", "QT", "TD", "QTD", "U", "QU", "FF", "QFF", "DD", "QDD", "RR1", "QRR1",
            "GLO", "QGLO", "N", "QN", "PMER", "QPMER"]
DOSSIER = config.dossier("stations")


def catalogue(rafraichir: bool = False) -> pd.DataFrame:
    """Liste des fichiers : departement, periode, url, taille_mo (mise en cache)."""
    cache = DOSSIER / "_catalogue.json"
    if cache.exists() and not rafraichir:
        return pd.DataFrame(json.loads(cache.read_text()))
    lignes, url = [], f"{API}/2/datasets/{JEU_ID}/resources/?page_size=200"
    while url:
        page = get(url).json()
        for res in page.get("data", []):
            lien = res.get("url") or ""
            m = NOM_FICHIER.search(lien.rsplit("/", 1)[-1]) or NOM_FICHIER.search(res.get("title", ""))
            if m:
                lignes.append({"departement": m.group(1), "periode": m.group(2), "url": lien,
                               "taille_mo": round((res.get("filesize") or 0) / 1e6, 1)})
        url = page.get("next_page")
    cache.write_text(json.dumps(lignes, indent=1))
    return pd.DataFrame(lignes)


def chemin(departement: str, periode: str):
    return DOSSIER / f"H_{departement}_{periode}.parquet"


def telecharger(departement: str, periode: str = "latest") -> list:
    """Télécharge les fichiers du département dont la période contient `periode`.

    Exemples : periode="latest" (mois récents), "2024" (fichiers couvrant 2024), "" (tout).
    Renvoie la liste des fichiers Parquet (déjà présents ou nouveaux).
    """
    cat = catalogue()
    choix = cat[(cat["departement"] == departement) & cat["periode"].str.contains(periode)]
    if choix.empty:
        raise ValueError(f"aucun fichier pour le département {departement} et la période « {periode} »")
    sorties = []
    for f in choix.itertuples():
        out = chemin(departement, f.periode)
        if not out.exists():
            print(f"  stations {departement} {f.periode} ({f.taille_mo} Mo)")
            brut = DOSSIER / f"H_{departement}_{f.periode}.csv.gz"
            telecharger_fichier(f.url, brut)
            df = pd.read_csv(brut, sep=";", usecols=lambda c: c.strip() in COLONNES,
                             dtype={"NUM_POSTE": str, "NOM_USUEL": str}, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            df["time"] = pd.to_datetime(df.pop("AAAAMMJJHH").astype("Int64").astype(str),
                                        format="%Y%m%d%H", errors="coerce")
            df.dropna(subset=["time"]).to_parquet(out, index=False)
            brut.unlink()
        sorties.append(out)
    return sorties


def lire(departement: str) -> pd.DataFrame:
    """Toutes les mesures téléchargées du département, horodatage ramené en UTC."""
    fichiers = sorted(DOSSIER.glob(f"H_{departement}_*.parquet"))
    if not fichiers:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fichiers], ignore_index=True)
    df = df.drop_duplicates(["NUM_POSTE", "time"]).sort_values(["NUM_POSTE", "time"])
    if departement in config.DECALAGE_UTC:            # outre-mer : heure locale -> UTC
        df["time"] = df["time"] - pd.Timedelta(hours=config.decalage_utc(departement))
    return df.reset_index(drop=True)


def stations(mesures: pd.DataFrame, part_min_T: float = 0.5) -> pd.DataFrame:
    """Une ligne par station qui mesure T sur au moins `part_min_T` des heures."""
    g = mesures.groupby("NUM_POSTE")
    st = g.agg(nom=("NOM_USUEL", "first"), lat=("LAT", "first"), lon=("LON", "first"),
               alti=("ALTI", "first"), debut=("time", "min"), fin=("time", "max"))
    st["part_T"] = g["T"].apply(lambda s: s.notna().mean())
    return st[st["part_T"] >= part_min_T].reset_index()


def temperature(mesures: pd.DataFrame, station: str, qualites=(0, 1)) -> pd.Series:
    """T horaire (°C) d'une station, en ne gardant que les qualités demandées (indexée en UTC)."""
    s = mesures[mesures["NUM_POSTE"] == station].set_index("time").sort_index()
    t = s["T"].where(s["QT"].isin(qualites)) if "QT" in s else s["T"]
    return t[~t.index.duplicated()].astype("float32")


if __name__ == "__main__":
    dep = sys.argv[1] if len(sys.argv) > 1 else "13"
    telecharger(dep, "latest")
    m = lire(dep)
    print(stations(m).to_string())
