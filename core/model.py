"""Modèle : jeu d'entraînement multi-stations, gradient boosting, validation honnête.

Cible : écart entre la température mesurée à la station et ERA5 au même point
(T_cible - era5_T). Le modèle apprend l'effet local (ville, mer, relief) à partir
de la météo de fond (ERA5 aujourd'hui, AROME en prévision) et de la fiche du site.

Validation :
- **stations exclues** (GroupKFold) : chaque station est prédite par un modèle qui
  ne l'a jamais vue, comme un nouveau site client ;
- **split temporel** : entraînement avant une date, test après (toutes stations).
Référence : ERA5 brut, et ERA5 corrigé du biais moyen appris sur l'entraînement.

Algorithme : LightGBM si disponible (sur Mac : `brew install libomp`), sinon
HistGradientBoostingRegressor de scikit-learn (même famille d'algorithme).

En ligne de commande (long : plusieurs minutes par station à la première préparation) :
    python -m core.model --zone 13 --debut 2025-01-01 --fin 2025-12-31 --preparer --entrainer
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core import features, zones
from core.config import DATA_DIR, PROCESSED_DIR
from core.sources import bdnb, era5, landsat, sentinel2
from core.sources import cerema_lcz as lcz
from core.sources import meteofrance_stations as mfs

PROC = PROCESSED_DIR / "modele"
MODELS_DIR = DATA_DIR / "modeles"
PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ERA5_DATASET = "era5-land"
EXCLUDE = {"lat", "lon", "rayon_m", "rayon_sat_m", "T_cible", "station", "cible", "time_utc"}
HOT = 30.0  # °C : seuil des « heures chaudes » dans les scores


# --------------------------------------------------------------------------- #
# Algorithme
# --------------------------------------------------------------------------- #
def backend() -> str:
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except Exception:  # absent, ou libomp manquant sur Mac
        return "sklearn"


def _new_model(kind: str, seed: int = 0):
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
                                         categorical_features="from_dtype")


# --------------------------------------------------------------------------- #
# Stations et préparation des données par station
# --------------------------------------------------------------------------- #
def stations_table(zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(observations, stations avec température sur plus de la moitié des heures)."""
    per = mfs.downloaded_periods(zone)
    if not per:
        return pd.DataFrame(), pd.DataFrame()
    obs = mfs.load(zone, per)
    st = mfs.stations_summary(obs)
    st = st[st.get("part_T", 0) > 0.5].copy()
    st["lat"], st["lon"] = st["lat"].astype(float).round(6), st["lon"].astype(float).round(6)
    return obs, st.reset_index(drop=True)


def _era5_covers(lat: float, lon: float, start: str, end: str) -> bool:
    if not era5.is_ready(ERA5_DATASET, lat, lon):
        return False
    idx = era5.load(ERA5_DATASET, lat, lon).index
    return idx.min() <= pd.Timestamp(start) and idx.max() >= pd.Timestamp(end) + pd.Timedelta(hours=23)


def readiness(stations: pd.DataFrame, zone: str, start: str, end: str, radius: int, radius_sat: int,
              lcz_catalog=None) -> pd.DataFrame:
    """Ce qui est déjà en local pour chaque station."""
    year = features.last_summer()
    done_lcz = set(lcz.downloaded())
    rows = []
    for r in stations.itertuples():
        terr = lcz.territory_at(r.lat, r.lon, lcz_catalog) if lcz_catalog is not None and len(lcz_catalog) else None
        rows.append({
            "station": r.NUM_POSTE, "nom": r.nom,
            "ERA5": "✅" if _era5_covers(r.lat, r.lon, start, end) else "❌",
            "BDNB": "–" if zones.is_outre_mer(zone) else ("✅" if bdnb.is_ready(r.lat, r.lon, radius) else "❌"),
            "Sentinel-2": "✅" if sentinel2.cache_path(r.lat, r.lon, radius, year).exists() else "❌",
            "Landsat": "✅" if landsat.is_ready(r.lat, r.lon, radius_sat) else "❌",
            "LCZ": "–" if terr is None else ("✅" if terr in done_lcz else f"❌ {terr}"),
        })
    return pd.DataFrame(rows)


def prepare_station(row, zone: str, start: str, end: str, radius: int, radius_sat: int, steps: set,
                    lcz_catalog=None, log=print) -> list[str]:
    """Télécharge ce qui manque pour une station ; renvoie la liste des erreurs (non bloquantes)."""
    lat, lon, errors = row.lat, row.lon, []

    def attempt(name, fn):
        try:
            fn()
        except Exception as exc:
            errors.append(f"{row.NUM_POSTE} {name} : {exc}")
            log(f"  {name} : échec ({exc})")

    if "era5" in steps and not _era5_covers(lat, lon, start, end):
        log(f"  ERA5-Land {start} → {end}")
        attempt("ERA5", lambda: era5.fetch(ERA5_DATASET, lat, lon, start, end))
    if "bdnb" in steps and not zones.is_outre_mer(zone) and not bdnb.is_ready(lat, lon, radius):
        log(f"  BDNB {radius} m")
        attempt("BDNB", lambda: bdnb.fetch(lat, lon, radius))
    if "s2" in steps:
        attempt("Sentinel-2", lambda: sentinel2.vegetation(lat, lon, radius, features.last_summer()))
    if "landsat" in steps and not landsat.is_ready(lat, lon, radius_sat):
        y = features.last_summer()
        log(f"  Landsat étés {y - 1}-{y}")
        attempt("Landsat", lambda: landsat.fetch(lat, lon, radius_sat, [y - 1, y], [6, 7, 8]))
    if "lcz" in steps and lcz_catalog is not None and len(lcz_catalog):
        terr = lcz.territory_at(lat, lon, lcz_catalog)
        if terr is not None and terr not in lcz.downloaded():
            url = lcz_catalog.loc[lcz_catalog["code"] == terr, "url"].iloc[0]
            log(f"  LCZ territoire {terr}")
            attempt("LCZ", lambda: lcz.fetch(terr, url))
    return errors


# --------------------------------------------------------------------------- #
# Jeu d'entraînement
# --------------------------------------------------------------------------- #
def dataset_path(zone: str) -> Path:
    return PROC / f"jeu_{zone}.parquet"


def build_dataset(zone: str, obs: pd.DataFrame, stations: pd.DataFrame, start: str, end: str,
                  radius: int, radius_sat: int, lcz_catalog=None, log=print) -> pd.DataFrame:
    """Une ligne par station et par heure : cible, ERA5, heure, fiche du site."""
    frames = []
    for r in stations.itertuples():
        site = {"lat": r.lat, "lon": r.lon, "label": r.nom}
        fiche, _ = features.static_features(site, zone, radius, radius_sat, lcz_catalog)
        table = features.hourly_table(site, zone, obs, None, r.NUM_POSTE)
        if "era5_T" not in table or "T_cible" not in table:
            log(f"  {r.nom} : ignorée (pas d'ERA5 ou pas de mesure)")
            continue
        table = table.loc[start:f"{end} 23:00"].dropna(subset=["T_cible", "era5_T"])
        if table.empty:
            log(f"  {r.nom} : ignorée (aucune heure commune sur la période)")
            continue
        frames.append(features.with_static(table, fiche).assign(station=r.NUM_POSTE, nom=r.nom))
        log(f"  {r.nom} : {len(table)} heures")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).reset_index()
    df.to_parquet(dataset_path(zone), index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Variables explicatives : ERA5, heure, fiche (sans identifiants ni cible)."""
    cols = []
    for c in df.columns:
        if c in EXCLUDE or c in ("nom",) or c.startswith("ref_"):
            continue
        if c == "lcz_site" or pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().any() and df[c].nunique(dropna=True) > 1:
                cols.append(c)
    return cols


def _matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    if "lcz_site" in X:
        X["lcz_site"] = pd.Categorical(X["lcz_site"].astype("string"), categories=list(lcz.LCZ_CLASSES))
    for c in X.columns:
        if c != "lcz_site":
            X[c] = X[c].astype("float64")
    return X


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def _scores(err: pd.Series) -> dict:
    err = err.dropna()
    return {"n": int(len(err)), "biais": float(err.mean()), "MAE": float(err.abs().mean()),
            "RMSE": float(np.sqrt((err ** 2).mean()))} if len(err) else {"n": 0}


def score_table(df: pd.DataFrame, pred: pd.Series, bias: pd.Series) -> pd.DataFrame:
    """Scores ERA5 brut / ERA5 + biais moyen / modèle : global, heures chaudes, Tmin et Tmax journalières."""
    out = []
    preds = {"ERA5 brut": df["era5_T"], "ERA5 + biais moyen": df["era5_T"] + bias, "modèle": pred}
    hot = df["T_cible"] >= HOT
    day = df["time_utc"].dt.floor("D") if "time_utc" in df else df.index.floor("D")
    for name, p in preds.items():
        e = p - df["T_cible"]
        daily = pd.DataFrame({"p": p, "o": df["T_cible"], "s": df["station"], "d": day}).groupby(["s", "d"])
        dmin = daily["p"].min() - daily["o"].min()
        dmax = daily["p"].max() - daily["o"].max()
        for scope, err in (("toutes les heures", e), (f"heures ≥ {HOT:.0f} °C", e[hot]),
                           ("Tmin journalière", dmin), ("Tmax journalière", dmax)):
            out.append({"prédicteur": name, "périmètre": scope, **_scores(err)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Entraînement et validation
# --------------------------------------------------------------------------- #
def cross_validate(df: pd.DataFrame, n_folds: int = 5, kind: str | None = None, log=print) -> dict:
    """Validation par stations exclues : prédictions hors échantillon pour toutes les stations."""
    from sklearn.model_selection import GroupKFold

    kind = kind or backend()
    cols = feature_columns(df)
    X, y = _matrix(df, cols), df["T_cible"] - df["era5_T"]
    groups = df["station"]
    n_folds = max(2, min(n_folds, groups.nunique()))
    oof, bias = pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = _new_model(kind, seed=k)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict(X.iloc[te])
        bias.iloc[te] = y.iloc[tr].mean()
        log(f"  pli {k + 1}/{n_folds} : stations test {sorted(groups.iloc[te].unique())}")
    pred = df["era5_T"] + oof
    per_station = []
    for s, g in df.assign(pred=pred, bias=bias).groupby("station"):
        per_station.append({"station": s, "nom": g["nom"].iloc[0], "heures": len(g),
                            "RMSE ERA5": _scores(g["era5_T"] - g["T_cible"])["RMSE"],
                            "RMSE modèle": _scores(g["pred"] - g["T_cible"])["RMSE"]})
    return {"kind": kind, "features": cols, "scores": score_table(df, pred, bias),
            "par_station": pd.DataFrame(per_station), "pred": pred}


def temporal_holdout(df: pd.DataFrame, split: str, kind: str | None = None) -> dict | None:
    """Entraînement avant `split`, test après (toutes stations)."""
    kind = kind or backend()
    t = df["time_utc"]
    tr, te = df[t < pd.Timestamp(split)], df[t >= pd.Timestamp(split)]
    if len(tr) < 1000 or len(te) < 100:
        return None
    cols = feature_columns(tr)
    m = _new_model(kind)
    m.fit(_matrix(tr, cols), tr["T_cible"] - tr["era5_T"])
    pred = te["era5_T"] + m.predict(_matrix(te, cols))
    bias = pd.Series((tr["T_cible"] - tr["era5_T"]).mean(), index=te.index)
    return {"split": split, "scores": score_table(te, pd.Series(pred, index=te.index), bias)}


def fit_final(df: pd.DataFrame, zone: str, meta: dict, kind: str | None = None) -> Path:
    """Modèle final sur toutes les données, sauvegardé avec ses métadonnées."""
    kind = kind or backend()
    cols = feature_columns(df)
    m = _new_model(kind)
    m.fit(_matrix(df, cols), df["T_cible"] - df["era5_T"])
    importance = None
    if kind == "lightgbm":
        imp = m.booster_.feature_importance(importance_type="gain")
        importance = dict(sorted(zip(cols, (imp / imp.sum()).round(4).tolist()), key=lambda x: -x[1]))
    path = MODELS_DIR / f"modele_{zone}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": m, "kind": kind, "features": cols, "zone": zone,
                     "entraine_le": dt.datetime.now().isoformat(timespec="seconds"), **meta}, f)
    path.with_suffix(".json").write_text(json.dumps(
        {"kind": kind, "features": cols, "zone": zone, "importance": importance, **meta},
        ensure_ascii=False, indent=2, default=str))
    return path


def load_model(zone: str) -> dict | None:
    path = MODELS_DIR / f"modele_{zone}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle: dict, table_with_static: pd.DataFrame) -> pd.Series:
    """Température prévue au site = ERA5 + écart prédit (NaN si ERA5 manque)."""
    X = _matrix(table_with_static.reindex(columns=bundle["features"]), bundle["features"])
    return table_with_static["era5_T"] + bundle["model"].predict(X)


# --------------------------------------------------------------------------- #
# Ligne de commande
# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - outil manuel
    p = argparse.ArgumentParser(description="Préparer les données et entraîner le modèle Hélio.")
    p.add_argument("--zone", required=True)
    p.add_argument("--debut", required=True, help="AAAA-MM-JJ")
    p.add_argument("--fin", required=True, help="AAAA-MM-JJ")
    p.add_argument("--rayon", type=int, default=150)
    p.add_argument("--rayon-sat", type=int, default=1000)
    p.add_argument("--sources", default="era5,bdnb,s2", help="parmi era5,bdnb,s2,landsat,lcz")
    p.add_argument("--preparer", action="store_true")
    p.add_argument("--entrainer", action="store_true")
    a = p.parse_args()
    obs, st = stations_table(a.zone)
    if st.empty:
        raise SystemExit("Aucune station : télécharge les observations dans l'onglet Stations.")
    try:
        cat = lcz.get_catalog()
    except Exception:
        cat = None
    if a.preparer:
        for i, r in enumerate(st.itertuples(), 1):
            print(f"[{i}/{len(st)}] {r.nom}")
            prepare_station(r, a.zone, a.debut, a.fin, a.rayon, a.rayon_sat, set(a.sources.split(",")), cat)
    if a.entrainer:
        df = build_dataset(a.zone, obs, st, a.debut, a.fin, a.rayon, a.rayon_sat, cat)
        if df.empty:
            raise SystemExit("Jeu d'entraînement vide (ERA5 manquant ?).")
        cv = cross_validate(df)
        print(cv["scores"].round(2).to_string())
        path = fit_final(df, a.zone, {"debut": a.debut, "fin": a.fin, "rayon": a.rayon,
                                      "stations": int(df["station"].nunique()), "heures": int(len(df))})
        print("Modèle enregistré :", path)


if __name__ == "__main__":
    main()
