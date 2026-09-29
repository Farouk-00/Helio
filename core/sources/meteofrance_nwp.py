"""Prévisions AROME / ARPEGE au site – API ciblée « Modèles » de Météo-France (WCS 2.0.1).

Chaîne : GetCapabilities (liste des couvertures : paramètre x niveau x run) ->
DescribeCoverage (échéances réellement disponibles du run) -> GetCoverage en
GeoTIFF sur une petite fenêtre autour du site, une requête par échéance et par
variable (50 requêtes / minute) -> valeur au point de grille le plus proche ->
Parquet par modèle et par site, qui s'enrichit à chaque run (archive des prévisions).

Clés : variables d'environnement METEOFRANCE_AROME_KEY et METEOFRANCE_ARPEGE_KEY
(en-tête HTTP « apikey »). Les identifiants de couverture ne sont pas codés en dur :
ils sont retrouvés dans GetCapabilities par leur préfixe.

Diagnostic (à lancer en local, n'affiche jamais la clé) :
    python -m core.sources.meteofrance_nwp
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.config import HTTP_TIMEOUT, PROCESSED_DIR

PROC = PROCESSED_DIR / "prevision"
PROC.mkdir(parents=True, exist_ok=True)

API = "https://public-api.meteofrance.fr/public"
MODELS = {
    "arome": {"libelle": "AROME – 0,01° (~1,3 km), France, jusqu'à ~48 h",
              "base": f"{API}/arome/1.0/wcs/MF-NWP-HIGHRES-AROME-001-FRANCE-WCS",
              "env": "METEOFRANCE_AROME_KEY", "grille": 0.01},
    "arpege": {"libelle": "ARPEGE – 0,1° (~10 km), Europe, jusqu'à ~4 jours",
               "base": f"{API}/arpege/1.0/wcs/MF-NWP-GLOBAL-ARPEGE-01-EUROPE-WCS",
               "env": "METEOFRANCE_ARPEGE_KEY", "grille": 0.1},
}
# Variable Hélio -> (préfixe de l'identifiant de couverture, hauteur (m) ou None, cumul horaire ?)
VARIABLES = {
    "T": ("TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 2, False),
    "RH": ("RELATIVE_HUMIDITY__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 2, False),
    "WS": ("WIND_SPEED__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", 10, False),
    "GHI": ("DOWNWARD_SHORT_WAVE_RADIATION_FLUX__GROUND_OR_WATER_SURFACE", None, True),
    "RR": ("TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE", None, True),
}
LIBELLES = {"T": "Température à 2 m (°C)", "RH": "Humidité relative à 2 m (%)", "WS": "Vent à 10 m (m/s)",
            "GHI": "Rayonnement solaire (W/m², moyenne de l'heure)", "RR": "Pluie (mm/h)"}
PAUSE_S = 1.3            # 50 requêtes / minute au maximum
FENETRE_PIX = 2          # ± 2 mailles autour du site


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def key(model: str) -> str | None:
    return os.environ.get(MODELS[model]["env"]) or None


def _get(model: str, op: str, params: dict, retries: int = 4) -> requests.Response:
    k = key(model)
    if not k:
        raise RuntimeError(f"clé absente : variable d'environnement {MODELS[model]['env']}")
    url = f"{MODELS[model]['base']}/{op}"
    for attempt in range(retries + 1):
        r = requests.get(url, params={"service": "WCS", "version": "2.0.1", **params},
                         headers={"apikey": k}, timeout=HTTP_TIMEOUT)
        if r.status_code == 429 or r.status_code >= 500:  # quota ou incident : on attend
            if attempt == retries:
                break
            time.sleep(float(r.headers.get("retry-after", 0) or 0) or (5 * 2 ** attempt))
            continue
        break
    if r.status_code != 200:
        raise RuntimeError(f"{op} : HTTP {r.status_code} – {r.text[:300]}")
    return r


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# --------------------------------------------------------------------------- #
# Catalogue : couvertures et runs
# --------------------------------------------------------------------------- #
_RUN_RE = re.compile(r"___(\d{4}-\d{2}-\d{2}T\d{2}[.:]\d{2}[.:]\d{2}Z)(.*)$")


def parse_coverage_id(cid: str) -> dict | None:
    """'TEMPERATURE__…___2026-09-29T00.00.00Z' -> {param, run (Timestamp UTC), suffix}."""
    m = _RUN_RE.search(cid)
    if not m:
        return None
    run = pd.Timestamp(m.group(1).replace(".", ":")).tz_convert(None)
    return {"id": cid, "param": cid[:m.start()], "run": run, "suffix": m.group(2).lstrip("_")}


def capabilities(model: str) -> pd.DataFrame:
    """Toutes les couvertures publiées : id, param, run, suffixe (ex. PT1H)."""
    r = _get(model, "GetCapabilities", {"language": "eng"})
    root = ET.fromstring(r.content)
    ids = [el.text.strip() for el in root.iter() if _local(el.tag) == "CoverageId" and el.text]
    rows = [p for p in (parse_coverage_id(i) for i in ids) if p]
    return pd.DataFrame(rows, columns=["id", "param", "run", "suffix"])


def coverage_for(cat: pd.DataFrame, var: str, run: pd.Timestamp) -> str | None:
    """Identifiant de couverture d'une variable Hélio pour un run (cumul : suffixe PT1H)."""
    prefix, _, cumul = VARIABLES[var]
    sel = cat[(cat["param"] == prefix) & (cat["run"] == run)]
    if cumul:
        sel = sel[sel["suffix"].str.upper() == "PT1H"]
    else:
        sel = sel[sel["suffix"] == ""]
    return None if sel.empty else sel.iloc[0]["id"]


def runs(cat: pd.DataFrame) -> list[pd.Timestamp]:
    """Runs pour lesquels la température à 2 m est publiée, du plus récent au plus ancien."""
    t = cat[cat["param"] == VARIABLES["T"][0]]
    return sorted(t["run"].unique(), reverse=True)


# --------------------------------------------------------------------------- #
# Axes d'une couverture (échéances, hauteurs)
# --------------------------------------------------------------------------- #
def describe(model: str, coverage_id: str) -> dict:
    """{'times': [Timestamp UTC…], 'heights': [float…]} lus dans DescribeCoverage."""
    r = _get(model, "DescribeCoverage", {"coverageid": coverage_id})
    root = ET.fromstring(r.content)
    begin = next((el.text for el in root.iter() if _local(el.tag) == "beginPosition"), None)
    out: dict = {"times": [], "heights": []}
    for axis in (el for el in root.iter() if _local(el.tag) == "GeneralGridAxis"):
        coef = next((c.text for c in axis.iter() if _local(c.tag) == "coefficients"), "") or ""
        name = next((c.text for c in axis.iter() if _local(c.tag) == "gridAxesSpanned"), "") or ""
        values = coef.split()
        if name.strip().lower() == "time":
            try:  # décalages en secondes depuis le début de la période
                base = pd.Timestamp(begin).tz_convert(None) if begin else parse_coverage_id(coverage_id)["run"]
                out["times"] = [base + pd.Timedelta(seconds=float(v)) for v in values]
            except ValueError:  # ou dates ISO
                out["times"] = [pd.Timestamp(v.strip('"')).tz_convert(None) for v in values]
        elif name.strip().lower() == "height":
            out["heights"] = [float(v) for v in values]
    return out


# --------------------------------------------------------------------------- #
# Valeur au site
# --------------------------------------------------------------------------- #
def value_at(model: str, coverage_id: str, when: pd.Timestamp, lat: float, lon: float,
             height: float | None) -> float:
    """Valeur au point de grille le plus proche du site, pour une échéance."""
    from rasterio.io import MemoryFile

    d = MODELS[model]["grille"] * FENETRE_PIX
    params = [("coverageid", coverage_id), ("format", "image/tiff"),
              ("subset", f"time({when:%Y-%m-%dT%H:%M:%SZ})"),
              ("subset", f"lat({lat - d:.4f},{lat + d:.4f})"),
              ("subset", f"long({lon - d:.4f},{lon + d:.4f})")]
    if height is not None:
        params.append(("subset", f"height({height:g})"))
    k = key(model)
    if not k:
        raise RuntimeError(f"clé absente : variable d'environnement {MODELS[model]['env']}")
    url = f"{MODELS[model]['base']}/GetCoverage"
    for attempt in range(5):
        r = requests.get(url, params=[("service", "WCS"), ("version", "2.0.1")] + params,
                         headers={"apikey": k}, timeout=HTTP_TIMEOUT)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("retry-after", 0) or 0) or (5 * 2 ** attempt))
            continue
        break
    if r.status_code != 200 or not r.content[:4] in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"GetCoverage {coverage_id} {when} : HTTP {r.status_code} – {r.text[:300]}")
    with MemoryFile(r.content) as mem, mem.open() as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        row, col = src.index(lon, lat)
        row = min(max(row, 0), src.height - 1)
        col = min(max(col, 0), src.width - 1)
        v = arr[row, col]
    if nodata is not None and v == nodata:
        return np.nan
    return float(v)


def to_physical(var: str, values: pd.Series) -> pd.Series:
    """Unités de l'API -> unités Hélio (conversion décidée sur l'ordre de grandeur)."""
    if var == "T" and values.median() > 150:        # kelvins
        return values - 273.15
    if var == "RH" and values.max() <= 1.5:          # fraction
        return values * 100
    if var == "GHI":                                 # J/m² cumulés sur l'heure -> W/m² moyen
        return (values / 3600).clip(lower=0) if values.max() > 5000 else values.clip(lower=0)
    if var == "RR":
        return values.clip(lower=0)
    return values


# --------------------------------------------------------------------------- #
# Téléchargement d'un run au site + archive
# --------------------------------------------------------------------------- #
def site_path(model: str, lat: float, lon: float) -> Path:
    return PROC / f"{model}_{lat:.4f}_{lon:.4f}.parquet"


def plan(model: str, cat: pd.DataFrame, run: pd.Timestamp, variables: list[str], horizon_h: int
         ) -> list[tuple[str, str, pd.Timestamp, float | None]]:
    """Liste des requêtes (variable, couverture, échéance, hauteur) jusqu'à run + horizon."""
    todo = []
    for var in variables:
        cid = coverage_for(cat, var, run)
        if cid is None:
            continue
        axes = describe(model, cid)
        time.sleep(PAUSE_S)
        height = VARIABLES[var][1]
        if height is not None and axes["heights"] and height not in axes["heights"]:
            continue
        for t in axes["times"]:
            if run < t <= run + pd.Timedelta(hours=horizon_h) or (t == run and not VARIABLES[var][2]):
                todo.append((var, cid, t, height))
    return todo


def fetch(model: str, lat: float, lon: float, run: pd.Timestamp, variables: list[str], horizon_h: int,
          cat: pd.DataFrame, progress=None) -> Path:
    """Télécharge un run au site et l'ajoute à l'archive (run, échéance, variables)."""
    todo = plan(model, cat, run, variables, horizon_h)
    if not todo:
        raise RuntimeError("aucune échéance disponible pour ces variables et ce run")
    rows = []
    for i, (var, cid, t, h) in enumerate(todo):
        rows.append({"var": var, "valid_time": t, "value": value_at(model, cid, t, lat, lon, h)})
        if progress:
            progress((i + 1) / len(todo))
        time.sleep(PAUSE_S)
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="valid_time", columns="var", values="value")
    for var in wide.columns:
        wide[var] = to_physical(var, wide[var])
    wide = wide.reset_index()
    wide.insert(0, "run", run)
    wide["echeance_h"] = ((wide["valid_time"] - run) / pd.Timedelta(hours=1)).round().astype(int)
    out = site_path(model, lat, lon)
    if out.exists():
        old = pd.read_parquet(out)
        wide = pd.concat([old[old["run"] != run], wide], ignore_index=True)
    wide.sort_values(["run", "valid_time"]).to_parquet(out, index=False)
    return out


def load(model: str, lat: float, lon: float) -> pd.DataFrame:
    p = site_path(model, lat, lon)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Diagnostic en ligne de commande
# --------------------------------------------------------------------------- #
def _diag() -> None:  # pragma: no cover - outil manuel
    lat, lon = 43.3057, 5.3943  # Marseille-Observatoire
    for model in MODELS:
        print(f"===== {model} =====")
        if not key(model):
            print(f"clé absente ({MODELS[model]['env']})")
            continue
        try:
            cat = capabilities(model)
            print(f"{len(cat)} couvertures, {cat['param'].nunique()} paramètres, runs : "
                  + ", ".join(f"{r:%d/%m %Hh}" for r in runs(cat)[:6]))
            for var, (prefix, h, cumul) in VARIABLES.items():
                hits = cat[cat["param"].str.contains(prefix.split("__")[0], regex=False)]
                print(f"  {var}: {'OK' if (hits['param'] == prefix).any() else 'ABSENT'}  "
                      f"variantes : {sorted(hits['param'].unique())[:4]}  suffixes : {sorted(hits['suffix'].unique())[:4]}")
            run = runs(cat)[0]
            cid = coverage_for(cat, "T", run)
            axes = describe(model, cid)
            print(f"  {cid} : {len(axes['times'])} échéances "
                  f"({axes['times'][0] if axes['times'] else '?'} -> {axes['times'][-1] if axes['times'] else '?'}), "
                  f"hauteurs {axes['heights'][:6]}")
            t = axes["times"][min(12, len(axes["times"]) - 1)]
            v = value_at(model, cid, t, lat, lon, 2)
            print(f"  T2m brute à Marseille-Obs le {t} : {v}")
            gid = coverage_for(cat, "GHI", run)
            print(f"  rayonnement : {gid}")
        except Exception as exc:
            print("ERREUR :", exc)


if __name__ == "__main__":
    _diag()
