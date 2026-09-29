"""Onglet Modèle : préparer les stations, entraîner, valider, prédire au site."""
import datetime as dt
import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Hélio – Modèle", page_icon="🤖", layout="wide")

from core import features, model, ui  # noqa: E402
from core.sources import cerema_lcz as lcz  # noqa: E402
from core.sources import era5  # noqa: E402
from core.sources import meteofrance_stations as mfs  # noqa: E402
from core.sources import sentinel2  # noqa: E402

SOURCES = {"era5": "ERA5-Land (indispensable, ~1 min / station)", "bdnb": "BDNB (~5-60 s / station)",
           "s2": "Sentinel-2 végétation (~10 s / station)", "landsat": "Landsat (~1 min / station)",
           "lcz": "LCZ Cerema (zip de 20 à 230 Mo par territoire)"}


@st.cache_data(show_spinner="Chargement des observations…")
def stations(zone: str, periods: tuple):
    return model.stations_table(zone)


@st.cache_data(show_spinner=False)
def lcz_catalog():
    try:
        return lcz.get_catalog()
    except Exception:
        return None


@st.cache_data(show_spinner="Lecture du jeu d'entraînement…")
def load_dataset(zone: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(model.dataset_path(zone))


def results_path(zone: str):
    return model.PROC / f"validation_{zone}.json"


zone = ui.zone_selector()
st.title("🤖 Modèle – température au site")
st.caption("Cible : l'écart entre la température mesurée à une station et ERA5 au même point. Le modèle apprend "
           "l'effet local (ville, mer, relief) à partir de la météo de fond et de la fiche du site, puis "
           "s'applique à n'importe quel site. Validation sur des stations jamais vues à l'entraînement.")

obs, sta = stations(zone, tuple(mfs.downloaded_periods(zone)))
if sta.empty:
    st.info("Télécharge d'abord des observations dans l'onglet Stations pour cette zone.")
    st.stop()
cat = lcz_catalog()

# --------------------------------------------------------------------------- #
# 1. Données par station
# --------------------------------------------------------------------------- #
st.header("1. Données par station")
tmin, tmax = obs["time"].min().date(), obs["time"].max().date()
c1, c2, c3 = st.columns([2, 1, 1])
default_end = min(tmax, dt.date(tmax.year - 1, 12, 31)) if tmax.month < 12 else tmax
period = c1.date_input("Période d'entraînement", value=(max(tmin, dt.date(default_end.year, 1, 1)), default_end),
                       min_value=tmin, max_value=tmax, format="DD/MM/YYYY")
radius = c2.select_slider("Rayon urbain (m)", options=[50, 100, 150, 200, 300, 500], value=150)
radius_sat = c3.select_slider("Rayon satellite (m)", options=[500, 1000, 2000, 3000], value=1000)
if not (isinstance(period, tuple) and len(period) == 2):
    st.stop()
start, end = period[0].isoformat(), period[1].isoformat()

codes = st.multiselect("Stations", list(sta["NUM_POSTE"]), default=list(sta["NUM_POSTE"]),
                       format_func=dict(zip(sta["NUM_POSTE"], sta["nom"])).get)
chosen = sta[sta["NUM_POSTE"].isin(codes)]
steps = st.multiselect("Sources à préparer", list(SOURCES), default=["era5", "bdnb", "s2"], format_func=SOURCES.get)
st.dataframe(model.readiness(chosen, zone, start, end, radius, radius_sat, cat), hide_index=True)
if "era5" in steps and not era5.has_key():
    st.warning("Clé Copernicus CDS absente : ERA5 ne pourra pas être téléchargé (voir l'onglet ERA5).")
st.caption("La préparation peut être longue ; elle reprend là où elle s'est arrêtée. En terminal : "
           f"`python -m core.model --zone {zone} --debut {start} --fin {end} --rayon {radius} "
           f"--sources {','.join(steps) or 'era5'} --preparer`")
if st.button("Préparer les données manquantes", disabled=not (steps and len(chosen))):
    bar = st.progress(0.0)
    errors = []
    with st.status("Préparation en cours…", expanded=True) as status:
        for i, r in enumerate(chosen.itertuples(), 1):
            bar.progress((i - 1) / len(chosen), text=f"{r.nom} ({i}/{len(chosen)})")
            st.write(f"**{r.nom}**")
            errors += model.prepare_station(r, zone, start, end, radius, radius_sat, set(steps), cat, log=st.write)
        bar.progress(1.0, text="Terminé")
        status.update(label=f"Préparation terminée ({len(errors)} erreur(s))", state="complete" if not errors else "error")
    for e in errors:
        st.error(e)
    st.cache_data.clear()

# --------------------------------------------------------------------------- #
# 2. Jeu d'entraînement
# --------------------------------------------------------------------------- #
st.header("2. Jeu d'entraînement")
if st.button("Construire le jeu d'entraînement", type="primary", disabled=chosen.empty):
    with st.status("Assemblage station par station…", expanded=False) as status:
        df_new = model.build_dataset(zone, obs, chosen, start, end, radius, radius_sat, cat, log=st.write)
        status.update(label=f"Jeu construit : {len(df_new):,} lignes".replace(",", " "), state="complete")
    st.cache_data.clear()
    st.rerun()

path = model.dataset_path(zone)
if not path.exists():
    st.info("Aucun jeu d'entraînement pour cette zone : prépare les données puis construis le jeu.")
    st.stop()
df = load_dataset(zone, path.stat().st_mtime)
cols = model.feature_columns(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes (station × heure)", f"{len(df):,}".replace(",", " "))
k2.metric("Stations", df["station"].nunique())
k3.metric("Période", f"{df['time_utc'].min():%d/%m/%Y} → {df['time_utc'].max():%d/%m/%Y}")
k4.metric("Variables explicatives", len(cols))
with st.expander("Variables et taux de remplissage"):
    st.dataframe(df[cols].notna().mean().rename("part renseignée").map(lambda x: f"{x:.0%}"))

# --------------------------------------------------------------------------- #
# 3. Entraînement et validation
# --------------------------------------------------------------------------- #
st.header("3. Entraînement et validation")
kind = model.backend()
if kind == "sklearn":
    st.info("LightGBM indisponible : scikit-learn (HistGradientBoosting) est utilisé. Sur Mac, pour LightGBM : "
            "`brew install libomp` puis `pip install lightgbm`.")
c1, c2 = st.columns(2)
n_folds = c1.slider("Plis (groupes de stations exclues)", 3, 10, 5)
t0, t1 = df["time_utc"].min(), df["time_utc"].max()
split = c2.date_input("Date de coupure du test temporel", value=(t0 + (t1 - t0) * 0.75).date(),
                      min_value=t0.date(), max_value=t1.date(), format="DD/MM/YYYY")
if st.button("Entraîner et valider", type="primary"):
    with st.status(f"Validation par stations exclues ({kind})…", expanded=True) as status:
        cv = model.cross_validate(df, n_folds, kind, log=st.write)
        st.write("Test temporel…")
        th = model.temporal_holdout(df, split.isoformat(), kind)
        st.write("Test stations exclues + période future…")
        stt = model.spatiotemporal(df, split.isoformat(), n_folds, kind)
        st.write("Modèle final sur toutes les données…")
        e = cv["pred"] - df["T_cible"]
        hourly = pd.DataFrame({"modèle": e, "ERA5 brut": df["era5_T"] - df["T_cible"]}).groupby(df["heure_locale"]).mean()
        res = {"kind": cv["kind"], "scores": cv["scores"].to_dict("records"),
               "par_station": cv["par_station"].to_dict("records"),
               "par_heure": hourly.reset_index().to_dict("records"),
               "temporel": th["scores"].to_dict("records") if th else None,
               "spatiotemporel": stt["scores"].to_dict("records") if stt else None, "coupure": split.isoformat()}
        model.fit_final(df, zone, {"debut": str(df["time_utc"].min().date()), "fin": str(df["time_utc"].max().date()),
                                   "rayon": radius, "rayon_sat": radius_sat, "stations": int(df["station"].nunique()),
                                   "heures": int(len(df))}, kind)
        results_path(zone).write_text(json.dumps(res, ensure_ascii=False, default=float))
        status.update(label="Modèle entraîné et enregistré", state="complete")

if results_path(zone).exists():
    res = json.loads(results_path(zone).read_text())
    scores = pd.DataFrame(res["scores"])
    st.markdown(f"**Validation par stations exclues** ({res['kind']}) – RMSE en °C, erreur = prévu − mesuré")
    def rmse_table(records):
        t = pd.DataFrame(records).pivot(index="périmètre", columns="prédicteur", values="RMSE")
        return t[["ERA5 brut", "ERA5 + biais moyen", "modèle"]].style.format(precision=2, na_rep="–")

    st.dataframe(rmse_table(res["scores"]))
    with st.expander("Biais, MAE et effectifs"):
        st.dataframe(scores.round(2), hide_index=True)
    ps = pd.DataFrame(res["par_station"]).sort_values("RMSE ERA5")
    fig = px.bar(ps.melt(id_vars=["nom"], value_vars=["RMSE ERA5", "RMSE modèle"], var_name="série",
                         value_name="RMSE (°C)"),
                 x="nom", y="RMSE (°C)", color="série", barmode="group")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="")
    st.plotly_chart(fig)
    ph = pd.DataFrame(res["par_heure"]).melt(id_vars=["heure_locale"], var_name="prédicteur", value_name="biais (°C)")
    fig = px.line(ph, x="heure_locale", y="biais (°C)", color="prédicteur", markers=True)
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(dtick=2, title="heure locale normale"))
    st.plotly_chart(fig)
    if res.get("temporel"):
        st.markdown(f"**Test temporel** (entraînement avant le {res['coupure']}, test après ; stations connues) – RMSE °C")
        st.dataframe(rmse_table(res["temporel"]))
    if res.get("spatiotemporel"):
        st.markdown(f"**Test le plus exigeant : stations exclues ET période après le {res['coupure']}** "
                    "(la situation d'un nouveau client en prévision) – RMSE °C")
        st.dataframe(rmse_table(res["spatiotemporel"]))
        st.caption("« – » : aucune heure dans ce périmètre sur la période testée (par exemple ≥ 30 °C en automne). "
                   "Pour juger la canicule, placer la coupure avant un été.")
    meta_path = model.MODELS_DIR / f"modele_{zone}.json"
    if meta_path.exists():
        imp = json.loads(meta_path.read_text()).get("importance")
        if imp:
            s = pd.Series(imp).head(15)
            fig = px.bar(s[::-1], orientation="h", labels={"value": "part du gain", "index": ""})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.markdown("**Importance des variables** (modèle final, LightGBM)")
            st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# 4. Prédire au site
# --------------------------------------------------------------------------- #
st.header("4. Prédire au site")
bundle = model.load_model(zone)
site = st.session_state.get("site")
if bundle is None:
    st.info("Entraîne d'abord un modèle (étape 3).")
    st.stop()
if not site:
    st.info("Choisis un site (onglets Urbain, Satellite, ERA5 ou Croisement).")
    st.stop()
st.caption(f"Site : **{site['label']}** ({site['lat']:.5f}, {site['lon']:.5f}). Modèle entraîné le "
           f"{bundle['entraine_le'].replace('T', ' ')} sur {bundle.get('stations', '?')} stations.")
if features.era5_dataset(site["lat"], site["lon"]) is None:
    st.info("Télécharge ERA5 pour ce site (onglet ERA5) : c'est la météo de fond du modèle.")
    st.stop()
fiche, _ = features.static_features(site, zone, bundle.get("rayon", radius), bundle.get("rayon_sat", radius_sat), cat)
table = features.with_static(features.hourly_table(site, zone, None, None, None), fiche)
manque = [c for c in bundle["features"] if c not in table or table[c].isna().all()]
if any(c.startswith("s2_") for c in manque):
    if st.button("Calculer la végétation Sentinel-2 pour ce site (~15 s)"):
        with st.spinner("Sentinel-2…"):
            try:
                sentinel2.vegetation(site["lat"], site["lon"], bundle.get("rayon", radius), features.last_summer())
                st.rerun()
            except Exception as exc:
                st.error(f"Sentinel-2 : échec ({exc})")
if manque:
    st.warning("Variables absentes pour ce site (prédiction dégradée) : " + ", ".join(manque)
               + ". Télécharge les sources correspondantes (Urbain, Satellite, Croisement).")
table = table.assign(prevue=model.predict(bundle, table))
d0, d1 = table.index.min().date(), table.index.max().date()
debut, fin = st.slider("Période affichée", min_value=d0, max_value=d1,
                       value=(max(d0, d1 - dt.timedelta(days=14)), d1), format="DD/MM/YYYY")
view = table.loc[str(debut):f"{fin} 23:00"]
long = view[["era5_T", "prevue"]].rename(columns={"era5_T": "ERA5 (maille)", "prevue": "modèle (site)"}) \
    .reset_index().melt(id_vars="time_utc", var_name="série", value_name="T (°C)")
fig = px.line(long, x="time_utc", y="T (°C)", color="série", labels={"time_utc": "UTC"})
fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
st.plotly_chart(fig)
ecart = (table["prevue"] - table["era5_T"]).groupby(table["heure_locale"]).mean()
c1, c2 = st.columns(2)
c1.metric("Écart moyen site − ERA5", f"{(table['prevue'] - table['era5_T']).mean():+.2f} °C")
c2.metric("Écart moyen à 4 h (nuit)", f"{ecart.get(4, float('nan')):+.2f} °C")
st.caption("Écart = effet local estimé par le modèle (îlot de chaleur, mer, relief). La prévision J+1/J+2 "
           "s'obtiendra en remplaçant ERA5 par AROME comme météo de fond (prochaine étape).")
