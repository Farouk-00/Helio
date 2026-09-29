"""Onglet Croisement : fiche du site + table horaire (entrée du modèle)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Hélio – Croisement", page_icon="🔀", layout="wide")

from core import features, ui, zones  # noqa: E402
from core.sources import cerema_lcz as lcz  # noqa: E402
from core.sources import meteofrance_stations as mfs  # noqa: E402
from core.sources import sentinel2  # noqa: E402


@st.cache_data(show_spinner="Chargement des observations…")
def load_stations(dep: str, periods: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = mfs.load(dep, list(periods))
    return df, mfs.stations_summary(df)


@st.cache_data(show_spinner=False)
def lcz_catalog():
    try:
        return lcz.get_catalog()
    except Exception:  # service Cerema indisponible : LCZ ignorées
        return None


@st.cache_resource(show_spinner="Chargement des LCZ du territoire…")
def load_lcz(code: str):
    return lcz.load(code)


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


zone = ui.zone_selector()
st.title("🔀 Croisement")
st.caption("Rassemble, pour un site, tout ce qui est en local : fiche fixe (LCZ, BDNB, Landsat, végétation, "
           "altitude, distance à la mer) et table horaire (station de référence, ERA5, cible). "
           "C'est la table d'entrée du modèle.")

radius = st.sidebar.select_slider("Rayon urbain (m)", options=[50, 100, 150, 200, 300, 500],
                                  value=st.session_state.get("rayon", 150),
                                  help="Même réglage que l'onglet Urbain (LCZ, BDNB, végétation).")
st.session_state["rayon"] = radius
radius_sat = st.sidebar.select_slider("Rayon satellite (m)", options=[500, 1000, 2000, 3000],
                                      value=st.session_state.get("rayon_sat", 1000),
                                      help="Même réglage que l'onglet Satellite.")
st.session_state["rayon_sat"] = radius_sat

# --------------------------------------------------------------------------- #
# Site : adresse libre, ou station Météo-France (cible connue)
# --------------------------------------------------------------------------- #
done = mfs.downloaded_periods(zone)
obs, stations = load_stations(zone, tuple(done)) if done else (None, pd.DataFrame())
with_t = stations[stations.get("part_T", 0) > 0.5] if not stations.empty else stations

mode = st.radio("Type de site", ["Station Météo-France (cible connue, pour entraîner le modèle)",
                                 "Site quelconque (adresse ou coordonnées)"])
target = None
if mode.startswith("Station"):
    if with_t.empty:
        st.info("Télécharge d'abord des observations dans l'onglet Stations pour cette zone.")
        st.stop()
    labels = {r.NUM_POSTE: f"{r.nom} ({r.NUM_POSTE}, {r.alti:.0f} m)" for r in with_t.itertuples()}
    current = st.session_state.get("cible")
    codes = list(with_t["NUM_POSTE"])
    target = st.selectbox("Station cible", codes, index=codes.index(current) if current in codes else 0,
                          format_func=labels.get)
    st.session_state["cible"] = target
    r = with_t.set_index("NUM_POSTE").loc[target]
    site = {"lat": round(float(r["lat"]), 6), "lon": round(float(r["lon"]), 6), "label": f"Station {r['nom']}"}
    st.session_state["site"] = site  # les onglets Urbain, Satellite, ERA5 suivent
    st.caption(f"Site = emplacement de la station ({site['lat']:.5f}, {site['lon']:.5f}). Les onglets Urbain, "
               "Satellite et ERA5 utilisent maintenant ce site : télécharge-y ses données si besoin.")
else:
    site = ui.site_selector()
    if not site:
        st.info("Choisis un site ci-dessus.")
        st.stop()

# --------------------------------------------------------------------------- #
# Sources disponibles
# --------------------------------------------------------------------------- #
st.subheader("Sources disponibles pour ce site")
cat = lcz_catalog()
st.dataframe(features.status(site, zone, radius, radius_sat, cat), hide_index=True)
year = features.last_summer()
if not sentinel2.cache_path(site["lat"], site["lon"], radius, year).exists():
    if st.button(f"Calculer la végétation Sentinel-2 (été {year}, ~15 s)"):
        with st.spinner("Sentinel-2 : lecture de la scène la moins nuageuse…"):
            try:
                sentinel2.vegetation(site["lat"], site["lon"], radius, year)
                st.rerun()
            except Exception as exc:
                st.error(f"Sentinel-2 : échec ({exc})")

# --------------------------------------------------------------------------- #
# Fiche statique
# --------------------------------------------------------------------------- #
st.subheader("Fiche du site")
fiche, warn = features.static_features(site, zone, radius, radius_sat, cat, lcz_loader=load_lcz)
for w in warn:
    st.warning(w)
shown = {k: v for k, v in fiche.items() if not (k.startswith("lcz_part_") and (pd.isna(v) or v == 0))}
c1, c2 = st.columns(2)
items = list(shown.items())
half = (len(items) + 1) // 2
for col, chunk in ((c1, items[:half]), (c2, items[half:])):
    col.dataframe(pd.DataFrame(chunk, columns=["variable", "valeur"]).astype({"valeur": str}), hide_index=True)
manquants = [k for k, v in fiche.items() if not k.startswith("lcz_part_") and (v is None or (isinstance(v, float) and np.isnan(v)))]
if manquants:
    st.caption(f"{len(manquants)} variables manquantes (source non téléchargée) : " + ", ".join(manquants))

# --------------------------------------------------------------------------- #
# Table horaire
# --------------------------------------------------------------------------- #
st.subheader("Table horaire")
ref = None
if not with_t.empty:
    cand = with_t[with_t["NUM_POSTE"] != target].copy()
    cand["km"] = distance_km(site["lat"], site["lon"], cand["lat"], cand["lon"])
    cand = cand.sort_values("km")
    opts = [None] + list(cand["NUM_POSTE"])
    ref = st.selectbox(
        "Station de référence (météo « de fond », hors site)", opts, index=1 if len(opts) > 1 else 0,
        format_func=lambda p: "aucune" if p is None else "{} ({:.1f} km, {:.0f} m)".format(
            *cand.set_index("NUM_POSTE").loc[p, ["nom", "km", "alti"]]),
        help="En production, la référence sera la prévision (AROME) ; ici, une station voisine ou ERA5.")

table = features.hourly_table(site, zone, obs, ref, target)
if table.empty:
    st.info("Aucune donnée horaire : télécharge des observations (Stations) ou ERA5 pour ce site.")
    st.stop()

k1, k2, k3 = st.columns(3)
k1.metric("Heures", f"{len(table):,}".replace(",", " "))
k2.metric("Période", f"{table.index.min():%d/%m/%Y} → {table.index.max():%d/%m/%Y}")
k3.metric("Avec cible", f"{int(table['T_cible'].notna().sum()):,}".replace(",", " ") if "T_cible" in table else "–",
          help="Heures où la température du site est mesurée (site = station).")

cover = table.notna().mean().rename("part renseignée").to_frame()
cover["part renseignée"] = cover["part renseignée"].map(lambda x: f"{x:.0%}")
with st.expander("Colonnes et taux de remplissage"):
    st.dataframe(cover)
st.dataframe(table.tail(48).round(2))

diag = features.diagnostics(table)
if not diag.empty:
    st.markdown("**Ce que le modèle devra corriger** : erreur des prédicteurs bruts contre la température du site")
    st.dataframe(diag, hide_index=True)
    local_cols = [c for c in ("ref_T", "era5_T") if c in table]
    prof = pd.DataFrame({f"cible − {c}": table["T_cible"] - table[c] for c in local_cols})
    prof = prof.groupby(table["heure_locale"]).mean().reset_index().melt(
        id_vars="heure_locale", var_name="écart", value_name="°C")
    fig = px.line(prof, x="heure_locale", y="°C", color="écart", markers=True,
                  labels={"heure_locale": f"heure locale normale (UTC{zones.utc_offset(zone):+g})"})
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title="", xaxis=dict(dtick=2))
    st.plotly_chart(fig)
    st.caption("Écart moyen selon l'heure entre le site et chaque prédicteur : c'est l'effet local "
               "(îlot de chaleur, mer, relief) que le modèle apprendra à partir de la fiche du site.")

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
full = features.with_static(table, fiche)
c1, c2 = st.columns(2)
if c1.button("Enregistrer pour le modèle", type="primary"):
    path = features.save(site, fiche, table)
    st.success(f"Enregistré : {path.name} (+ fiche JSON) dans data/processed/croisement/")
c2.download_button("Exporter en CSV (table + fiche)", full.reset_index().to_csv(index=False).encode("utf-8"),
                   file_name=f"croisement_{features.site_key(site['lat'], site['lon'])}.csv", mime="text/csv")
