"""Étape 1 : construire le jeu d'entraînement, une ligne par station et par heure.

Pour chaque département :
  1. mesures horaires des stations Météo-France (api_meteofrance_stations) ;
  2. pour chaque station qui mesure T : ERA5 au point (api_era5) et variables fixes du site
     (api_ign, api_cerema_lcz, api_bdnb, api_landsat, api_sentinel2) ;
  3. table horaire CONTINUE sur [DEBUT, FIN] (aucune heure sautée, pour que le modèle puisse
     lire les heures précédentes) ; T_station vaut NaN quand la mesure manque.
Sortie : donnees/jeu/jeu_<departement>.parquet

Tout est mis en cache source par source : relancer ne retélécharge rien, et une source qui
échoue pour une station laisse des NaN (message affiché) sans arrêter le reste.

Colonnes :
  station, nom, departement, lat, lon, time (UTC)
  T_station                 température mesurée (°C) ; la cible du modèle est T_station − era5_T
  era5_T, era5_TD, era5_RH, era5_P, era5_WS, era5_WD, era5_GHI, era5_IR, era5_RR, era5_TS
  heure_locale, jour_annee, cos_zenith
  altitude_m, tpi_500, tpi_2000, alt_maille, ecart_alt_maille, dist_mer_km   (IGN)
  lcz_site_*, lcz_part_*, lcz_hre…                                         (Cerema)
  bdnb_*                                                                   (BDNB)
  lst_site, lst_zone, lst_ecart, lst_rang, lst_dates                       (Landsat)
  s2_arbres, s2_herbe, s2_ndvi                                             (Sentinel-2)

Exemples :
    python construire_jeu.py                       # départements et période de config.py
    python construire_jeu.py --dep 13 84 --sans landsat lcz
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import api_bdnb
import api_cerema_lcz
import api_era5
import api_ign
import api_landsat
import api_meteofrance_stations as stations_mf
import api_sentinel2
import config
from outils import cos_zenith

SOURCES = ["ign", "lcz", "bdnb", "landsat", "sentinel2"]


def variables_fixes(lat: float, lon: float, departement: str, sources: list) -> dict:
    """Toutes les variables fixes d'un site ; une source en échec est simplement absente (NaN)."""
    grille = api_era5.JEUX[config.ERA5_JEU]["grille"]
    appels = {
        "ign": lambda: api_ign.caracteristiques(lat, lon, grille),
        "lcz": lambda: api_cerema_lcz.caracteristiques(lat, lon, config.RAYON_URBAIN_M),
        "bdnb": lambda: api_bdnb.caracteristiques(lat, lon, config.RAYON_URBAIN_M),
        "landsat": lambda: api_landsat.caracteristiques(lat, lon, config.RAYON_SATELLITE_M),
        "sentinel2": lambda: api_sentinel2.caracteristiques(lat, lon, config.RAYON_URBAIN_M),
    }
    if departement in config.DECALAGE_UTC:       # BDNB : métropole uniquement
        appels.pop("bdnb")
    fixes = {}
    for nom in sources:
        if nom in appels:
            try:
                fixes.update(appels[nom]())
            except Exception as exc:
                print(f"    {nom} : échec ({exc})")
    return {k: (np.nan if v is None else v) for k, v in fixes.items()}


def table_station(st, mesures: pd.DataFrame, departement: str, sources: list) -> pd.DataFrame | None:
    """Table horaire continue d'une station (None si ERA5 indisponible)."""
    try:
        api_era5.telecharger(st.lat, st.lon, config.DEBUT, config.FIN)
    except Exception as exc:
        print(f"    ERA5 : échec ({exc}) -> station ignorée")
        return None
    temps = pd.date_range(config.DEBUT, f"{config.FIN} 23:00", freq="h")
    e = api_era5.lire(st.lat, st.lon)
    t = pd.DataFrame(index=temps)
    t["T_station"] = stations_mf.temperature(mesures, st.NUM_POSTE).reindex(temps)
    t = t.join(e[~e.index.duplicated()].reindex(temps).add_prefix("era5_"))
    t["heure_locale"] = (temps + pd.Timedelta(hours=config.decalage_utc(departement))).hour
    t["jour_annee"] = temps.dayofyear
    t["cos_zenith"] = cos_zenith(temps, st.lat, st.lon)
    t = t.assign(**variables_fixes(st.lat, st.lon, departement, sources))
    t.insert(0, "station", st.NUM_POSTE)
    t.insert(1, "nom", st.nom)
    t.insert(2, "departement", departement)
    t.insert(3, "lat", st.lat)
    t.insert(4, "lon", st.lon)
    t.index.name = "time"
    return t.reset_index()


def construire(departement: str, sources: list = SOURCES):
    stations_mf.telecharger(departement, "latest")
    mesures = stations_mf.lire(departement)
    liste = stations_mf.stations(mesures)
    print(f"Département {departement} : {len(liste)} stations")
    tables = []
    for i, st in enumerate(liste.itertuples(), 1):
        print(f"  [{i}/{len(liste)}] {st.nom}")
        t = table_station(st, mesures, departement, sources)
        if t is not None and t["T_station"].notna().any():
            tables.append(t)
    if not tables:
        print("  aucune station exploitable")
        return None
    jeu = pd.concat(tables, ignore_index=True)
    jeu = jeu.astype({c: "float32" for c in jeu.select_dtypes("float64").columns})
    out = config.dossier("jeu") / f"jeu_{departement}.parquet"
    jeu.to_parquet(out, index=False)
    print(f"  -> {out} : {len(jeu):,} lignes, {jeu['T_station'].notna().sum():,} heures mesurées")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dep", nargs="+", default=config.DEPARTEMENTS, help="départements (défaut : config.py)")
    p.add_argument("--sans", nargs="*", default=[], help=f"sources à sauter parmi {SOURCES}")
    a = p.parse_args()
    for dep in a.dep:
        construire(dep, [s for s in SOURCES if s not in a.sans])
