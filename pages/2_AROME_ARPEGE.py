"""Onglet AROME / ARPEGE : prévisions Météo-France au site (API ciblée « Modèles »)."""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Hélio – AROME / ARPEGE", page_icon="🌦️", layout="wide")

from core import ui, zones  # noqa: E402
from core.sources import meteofrance_nwp as nwp  # noqa: E402


@st.cache_data(ttl=600, show_spinner="Lecture du catalogue Météo-France…")
def catalog(model: str) -> pd.DataFrame:
    return nwp.capabilities(model)


@st.cache_data(ttl=600, show_spinner=False)
def run_info(model: str, run: pd.Timestamp) -> dict:
    return nwp.run_info(model, catalog(model), run)


@st.cache_data(show_spinner=False)
def load(model: str, lat: float, lon: float, mtime: float) -> pd.DataFrame:  # mtime : relit après ajout
    return nwp.load(model, lat, lon)


zone = ui.zone_selector()
st.title("🌦️ AROME / ARPEGE – prévisions au site")
st.caption("Source : Météo-France, API ciblée « Modèles » (WCS). Point de grille le plus proche du site. "
           "Heures en **UTC**. Chaque run téléchargé est gardé : c'est l'archive des prévisions.")

if zones.is_outre_mer(zone):
    st.info("Outre-mer : AROME-OM (domaines Antilles, Guyane, océan Indien…) n'est pas encore branché ; "
            "seuls AROME France et ARPEGE Europe le sont.")

site = ui.site_selector()
if not site:
    st.info("Choisis un site ci-dessus (adresse ou coordonnées).")
    st.stop()
lat, lon = site["lat"], site["lon"]

model = st.radio("Modèle", list(nwp.MODELS), format_func=lambda m: nwp.MODELS[m]["libelle"], horizontal=True,
                 key="nwp_model")
if not nwp.key(model):
    st.warning(f"Clé absente : ajoute `export {nwp.MODELS[model]['env']}=\"…\"` dans `~/.zshrc`, "
               "puis relance Streamlit depuis un nouveau terminal.")
    st.stop()

# --------------------------------------------------------------------------- #
# Téléchargement d'un run
# --------------------------------------------------------------------------- #
try:
    cat = catalog(model)
    run_list = nwp.runs(cat)
except Exception as exc:
    st.error(f"Catalogue Météo-France inaccessible : {exc}")
    st.stop()
if not run_list:
    st.error("Aucun run trouvé dans le catalogue (température à 2 m absente).")
    st.stop()

archive = load(model, lat, lon, nwp.site_path(model, lat, lon).stat().st_mtime) \
    if nwp.site_path(model, lat, lon).exists() else pd.DataFrame()
have = set(pd.to_datetime(archive["run"]).unique()) if not archive.empty else set()

with st.expander("📥 Télécharger un run", expanded=archive.empty):
    c1, c2, c3 = st.columns([2, 2, 1])
    infos = {r: run_info(model, r) for r in run_list[:4]}  # 4 requêtes, gardées 10 min
    complet = next((i for i, r in enumerate(run_list[:4]) if infos[r]["n"] > 1), 0)
    run = c1.selectbox("Run", run_list, index=complet, format_func=lambda r: f"{r:%d/%m/%Y %H} h UTC"
                       + (f" – {infos[r]['n']} échéances" if r in infos else "")
                       + (" ✅ en local" if r in have else ""),
                       help="Le run le plus récent est publié progressivement : par défaut, le dernier run "
                            "qui a plus d'une échéance.")
    variables = c2.multiselect("Variables", list(nwp.VARIABLES), default=["T", "RH"],
                               format_func=nwp.LIBELLES.get)
    max_h = max(1, infos[run]["derniere_h"]) if run in infos else (51 if model == "arome" else 102)
    horizon = c3.number_input("Horizon (h)", 1, max_h, min(48, max_h))
    n_req = len(variables) * (horizon + 1)
    st.caption(f"Environ {n_req} requêtes, soit ~{n_req * nwp.PAUSE_S / 60:.0f} min (50 requêtes/minute au "
               "maximum). Le run le plus récent peut ne pas avoir encore toutes ses échéances.")
    missing = [v for v in variables if nwp.coverage_for(cat, v, run) is None]
    if missing:
        st.warning("Non publiées pour ce run : " + ", ".join(nwp.LIBELLES[v] for v in missing))
    if st.button("Télécharger", type="primary", disabled=not variables):
        bar = st.progress(0.0, text="Lecture des échéances disponibles…")
        try:
            nwp.fetch(model, lat, lon, run, variables, int(horizon), cat,
                      progress=lambda f: bar.progress(f, text=f"{model.upper()} : {f:.0%}"))
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"{model.upper()} : échec ({exc})")

if archive.empty:
    st.stop()

# --------------------------------------------------------------------------- #
# Prévision d'un run
# --------------------------------------------------------------------------- #
archive["run"] = pd.to_datetime(archive["run"])
runs_local = sorted(archive["run"].unique(), reverse=True)
st.subheader("Prévision")
shown_run = st.selectbox("Run affiché", runs_local, format_func=lambda r: f"{pd.Timestamp(r):%d/%m/%Y %H} h UTC")
fc = archive[archive["run"] == shown_run].set_index("valid_time").sort_index()
vars_here = [v for v in nwp.VARIABLES if v in fc and fc[v].notna().any()]

if "T" in fc and fc["T"].notna().any():
    k1, k2, k3 = st.columns(3)
    k1.metric("T max prévue", f"{fc['T'].max():.1f} °C", f"le {fc['T'].idxmax():%d/%m %H} h UTC", delta_color="off")
    k2.metric("T min prévue", f"{fc['T'].min():.1f} °C", f"le {fc['T'].idxmin():%d/%m %H} h UTC", delta_color="off")
    k3.metric("Échéances", f"{fc['echeance_h'].min()} → {fc['echeance_h'].max()} h")

for var in vars_here:
    fig = px.line(fc.reset_index(), x="valid_time", y=var, markers=True,
                  labels={"valid_time": "UTC", var: nwp.LIBELLES[var]})
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Comparaison des runs (même échéance, runs successifs) et des modèles
# --------------------------------------------------------------------------- #
if len(runs_local) > 1 and "T" in archive:
    st.subheader("Évolution des runs (température)")
    comp = archive.dropna(subset=["T"]).assign(run=lambda d: d["run"].dt.strftime("%d/%m %Hh"))
    fig = px.line(comp, x="valid_time", y="T", color="run", labels={"valid_time": "UTC", "T": "T 2 m (°C)"})
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.caption("Plus les runs successifs se ressemblent, plus la prévision est stable.")

other = "arpege" if model == "arome" else "arome"
other_df = nwp.load(other, lat, lon)
if not other_df.empty and "T" in other_df and "T" in fc:
    other_df["run"] = pd.to_datetime(other_df["run"])
    last_other = other_df[other_df["run"] == other_df["run"].max()]
    both = pd.concat([fc.reset_index()[["valid_time", "T"]].assign(modèle=f"{model.upper()} {pd.Timestamp(shown_run):%d/%m %Hh}"),
                      last_other[["valid_time", "T"]].assign(modèle=f"{other.upper()} {last_other['run'].max():%d/%m %Hh}")])
    st.subheader("AROME et ARPEGE")
    fig = px.line(both.dropna(), x="valid_time", y="T", color="modèle", labels={"valid_time": "UTC", "T": "T 2 m (°C)"})
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

with st.expander("Tableau et export"):
    st.dataframe(fc.round(2))
    st.download_button("Exporter en CSV (tous les runs en local)", archive.to_csv(index=False).encode("utf-8"),
                       file_name=f"{model}_{lat:.4f}_{lon:.4f}.csv", mime="text/csv")
