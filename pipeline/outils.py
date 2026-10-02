"""Petits outils partagés : requêtes HTTP avec reprises, téléchargement, position du soleil."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TIMEOUT = 120


def get(url: str, params=None, headers=None, essais: int = 4) -> requests.Response:
    """GET qui réessaie sur coupure réseau, 429 (quota) et erreurs 5xx."""
    for i in range(essais + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        except requests.RequestException:
            if i == essais:
                raise
            time.sleep(2 ** i)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if i == essais:
                r.raise_for_status()
            time.sleep(max(2 ** i, float(r.headers.get("retry-after", 0) or 0)))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("inatteignable")


def telecharger_fichier(url: str, dest: Path) -> Path:
    """Téléchargement en flux (gros fichiers), écrit d'abord dans un .part."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for bloc in r.iter_content(1 << 20):
                f.write(bloc)
    tmp.rename(dest)
    return dest


def cos_zenith(temps_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Cosinus de l'angle zénithal du soleil (formules NOAA, ~0,5°). > 0 : soleil levé."""
    doy = temps_utc.dayofyear.to_numpy()
    h = temps_utc.hour.to_numpy() + temps_utc.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (h - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    angle_horaire = np.radians((h * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(angle_horaire)


def humidite_relative(t_c, td_c):
    """Humidité relative (%) à partir de T et du point de rosée (Magnus)."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))
