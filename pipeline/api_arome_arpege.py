"""Météo-France – prévisions AROME / ARPEGE au site (API « Modèles », WCS 2.0.1).

Sert en prévision réelle : le modèle entraîné sur ERA5 prendra AROME comme météo de fond.
Les prévisions ne sont conservées que 5 jours par l'API : pour constituer une archive,
lancer ce script chaque jour (le Parquet du site s'enrichit à chaque run).

Clés : comptes sur portail-api.meteofrance.fr, puis dans ~/.zshrc
    export METEOFRANCE_AROME_KEY=...
    export METEOFRANCE_ARPEGE_KEY=...
(envoyées dans l'en-tête HTTP « apikey »). Quota : 50 requêtes / minute.

Fonctionnement : GetCapabilities (liste des couvertures « PARAMETRE___AAAA-MM-JJTHH.MM.SSZ »,
une par run ; suffixe _PT1H pour les cumuls horaires) -> DescribeCoverage (échéances du run)
-> GetCoverage en GeoTIFF sur une petite fenêtre, une requête par variable et par échéance.
AROME : 0,01°, runs toutes les 3 h, jusqu'à +51 h. ARPEGE : 0,1°, jusqu'à ~4 jours.

Exemple (dernier run complet, 24 h, au site) :
    python api_arome_arpege.py arome 43.3048 5.3955 24
"""
from __future__ import annotations

import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

import config

API = "https://public-api.meteofrance.fr/public"
MODELES = {
    "arome": {"base": f"{API}/arome/1.0/wcs/MF-NWP-HIGHRES-AROME-001-FRANCE-WCS",
              "cle": "METEOFRANCE_AROME_KEY", "grille": 0.01},
    "arpege": {"base": f"{API}/arpege/1.0/wcs/MF-NWP-GLOBAL-ARPEGE-01-EUROPE-WCS",
               "cle": "METEOFRANCE_ARPEGE_KEY", "grille": 0.1},
}
# variable -> (préfixe de la couverture, hauteur en m ou None, cumul horaire ?)
VARIABLES = {
    "T": ("TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 2, False),
    "RH": ("RELATIVE_HUMIDITY__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 2, False),
    "WS": ("WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 10, False),
    "GHI": ("DOWNWARD_SHORT_WAVE_RADIATION_FLUX__GROUND_OR_WATER_SURFACE", None, True),
    "RR": ("TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE", None, True),
}
PAUSE_S = 1.3
DOSSIER = config.dossier("prevision")
_RUN = re.compile(r"___(\d{4}-\d{2}-\d{2}T\d{2}[.:]\d{2}[.:]\d{2}Z)(.*)$")


def _requete(modele: str, operation: str, params: list) -> requests.Response:
    cle = os.environ.get(MODELES[modele]["cle"])
    if not cle:
        raise RuntimeError(f"clé absente : variable d'environnement {MODELES[modele]['cle']}")
    url = f"{MODELES[modele]['base']}/{operation}"
    for i in range(5):
        r = requests.get(url, params=[("service", "WCS"), ("version", "2.0.1")] + params,
                         headers={"apikey": cle}, timeout=120)
        if r.status_code != 429 and r.status_code < 500:
            break
        time.sleep(float(r.headers.get("retry-after", 0) or 0) or 5 * 2 ** i)
    if r.status_code != 200:
        raise RuntimeError(f"{operation} : HTTP {r.status_code} – {r.text[:300]}")
    time.sleep(PAUSE_S)
    return r


def _balise(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def couvertures(modele: str) -> pd.DataFrame:
    """Toutes les couvertures publiées : id, param, run (UTC), suffixe (ex. PT1H)."""
    racine = ET.fromstring(_requete(modele, "GetCapabilities", [("language", "eng")]).content)
    lignes = []
    for el in racine.iter():
        if _balise(el) == "CoverageId" and el.text and _RUN.search(el.text):
            m = _RUN.search(el.text)
            lignes.append({"id": el.text, "param": el.text[:m.start()],
                           "run": pd.Timestamp(m.group(1).replace(".", ":")).tz_convert(None),
                           "suffixe": m.group(2).lstrip("_")})
    return pd.DataFrame(lignes)


def echeances(modele: str, couverture: str) -> list:
    """Instants de validité (UTC) publiés pour une couverture."""
    racine = ET.fromstring(_requete(modele, "DescribeCoverage", [("coverageid", couverture)]).content)
    debut = next((el.text for el in racine.iter() if _balise(el) == "beginPosition"), None)
    for axe in (el for el in racine.iter() if _balise(el) == "GeneralGridAxis"):
        nom = next((c.text for c in axe.iter() if _balise(c) == "gridAxesSpanned"), "") or ""
        if nom.strip().lower() == "time":
            valeurs = (next((c.text for c in axe.iter() if _balise(c) == "coefficients"), "") or "").split()
            try:                                   # décalages en secondes depuis le début
                return [pd.Timestamp(debut).tz_convert(None) + pd.Timedelta(seconds=float(v)) for v in valeurs]
            except ValueError:                     # ou dates ISO
                return [pd.Timestamp(v.strip('"')).tz_convert(None) for v in valeurs]
    return []


def valeur(modele: str, couverture: str, quand: pd.Timestamp, lat: float, lon: float, hauteur=None) -> float:
    """Valeur au point de grille le plus proche du site, pour une échéance."""
    from rasterio.io import MemoryFile

    d = 2 * MODELES[modele]["grille"]
    params = [("coverageid", couverture), ("format", "image/tiff"),
              ("subset", f"time({quand:%Y-%m-%dT%H:%M:%SZ})"),
              ("subset", f"lat({lat - d:.4f},{lat + d:.4f})"), ("subset", f"long({lon - d:.4f},{lon + d:.4f})")]
    if hauteur is not None:
        params.append(("subset", f"height({hauteur:g})"))
    r = _requete(modele, "GetCoverage", params)
    with MemoryFile(r.content) as mem, mem.open() as src:
        ligne, col = src.index(lon, lat)
        v = float(src.read(1)[min(max(ligne, 0), src.height - 1), min(max(col, 0), src.width - 1)])
        return np.nan if src.nodata is not None and v == src.nodata else v


def prevision(modele: str, lat: float, lon: float, horizon_h: int = 48, run=None) -> pd.DataFrame:
    """Prévision au site (une ligne par échéance), unités : °C, %, m/s, W/m², mm/h.

    run=None : le run le plus récent dont la température est publiée sur tout l'horizon.
    """
    cat = couvertures(modele)
    t_cov = cat[(cat["param"] == VARIABLES["T"][0]) & (cat["suffixe"] == "")]
    if run is None:                                # runs du plus récent au plus ancien
        for r in sorted(t_cov["run"].unique(), reverse=True):
            ech = echeances(modele, t_cov.loc[t_cov["run"] == r, "id"].iloc[0])
            if ech and max(ech) >= r + pd.Timedelta(hours=horizon_h):
                run = r
                break
    lignes = []
    for var, (prefixe, hauteur, cumul) in VARIABLES.items():
        sel = cat[(cat["param"] == prefixe) & (cat["run"] == run)]
        sel = sel[sel["suffixe"].str.upper() == "PT1H"] if cumul else sel[sel["suffixe"] == ""]
        if sel.empty:
            continue
        cid = sel.iloc[0]["id"]
        for t in echeances(modele, cid):
            if run < t <= run + pd.Timedelta(hours=horizon_h):
                lignes.append({"var": var, "time": t, "valeur": valeur(modele, cid, t, lat, lon, hauteur)})
    p = pd.DataFrame(lignes).pivot_table(index="time", columns="var", values="valeur")
    if "T" in p and p["T"].median() > 150:
        p["T"] -= 273.15                           # kelvins -> °C
    if "GHI" in p and p["GHI"].max() > 5000:
        p["GHI"] = p["GHI"] / 3600                 # J/m² sur l'heure -> W/m² moyen
    if "GHI" in p:
        p["GHI"] = p["GHI"].clip(lower=0)
    if "RR" in p:
        p["RR"] = p["RR"].clip(lower=0)
    p.insert(0, "run", run)
    out = DOSSIER / f"{modele}_{lat:.4f}_{lon:.4f}.parquet"   # archive : on ajoute ce run
    archive = p.reset_index()
    if out.exists():
        ancien = pd.read_parquet(out)
        archive = pd.concat([ancien[ancien["run"] != run], archive], ignore_index=True)
    archive.to_parquet(out, index=False)
    return p


if __name__ == "__main__":
    modele = sys.argv[1] if len(sys.argv) > 1 else "arome"
    lat, lon = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (43.3048, 5.3955)
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    print(prevision(modele, lat, lon, h).round(1).to_string())
