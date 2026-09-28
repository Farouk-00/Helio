"""Onglet Satellite : température de surface Landsat 8/9 (été) autour d'un site."""
import datetime as dt

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hélio – Satellite", page_icon="🛰️", layout="wide")

from core import ui  # noqa: E402
from core.maps import colorbar_trace, disk_trace, layout_map, raster_layer, site_marker  # noqa: E402
from core.sources import landsat  # noqa: E402

MOIS = ["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"]


@st.cache_data(show_spinner="Chargement des données Landsat…")
def load(lat: float, lon: float, radius: int, mtime: float) -> dict:  # mtime : relit après mise à jour
    return landsat.load(lat, lon, radius)


zone = ui.zone_selector()
st.title("🛰️ Satellite – température de surface")
st.caption("Source : USGS Landsat 8/9, Collection 2 niveau 2 (bande thermique, 30 m), via Microsoft "
           "Planetary Computer. Température de la **surface** (toits, sol, végétation), pas de l'air ; "
           "passage vers 10 h 30 UTC, jamais de nuit.")

site = ui.site_selector()
if not site:
    st.info("Choisis un site ci-dessus (adresse ou coordonnées).")
    st.stop()
lat, lon = site["lat"], site["lon"]

# --------------------------------------------------------------------------- #
# Paramètres et téléchargement
# --------------------------------------------------------------------------- #
radius = st.sidebar.select_slider("Rayon de la zone satellite (m)", options=[500, 1000, 2000, 3000],
                                  value=st.session_state.get("rayon_sat", 1000),
                                  help="Zone de comparaison autour du site (écart site – zone).")
st.session_state["rayon_sat"] = radius

today = dt.date.today()
last = today.year if today.month >= 9 else today.year - 1   # dernier été complet
ready = landsat.is_ready(lat, lon, radius)
meta = load(lat, lon, radius, (landsat.site_dir(lat, lon, radius) / "meta.json").stat().st_mtime)["meta"] \
    if ready else None

with st.expander("📥 Scènes Landsat", expanded=not ready):
    c1, c2, c3 = st.columns([2, 2, 1])
    years = c1.multiselect("Étés", list(range(2013, last + 1))[::-1],
                           default=meta["annees"] if meta else [last - 2, last - 1, last],
                           help="Landsat 8 depuis 2013, Landsat 9 depuis 2022 (un passage tous les 8 jours à deux).")
    months = c2.multiselect("Mois", list(range(1, 13)), default=meta["mois"] if meta else [6, 7, 8],
                            format_func=lambda i: MOIS[i - 1])
    max_cloud = c3.number_input("Nuages max. (%)", 0, 100, int(meta["nuages_max"]) if meta else 30,
                                help="Couverture nuageuse de la scène entière ; les nuages sur la zone "
                                     "sont ensuite masqués pixel par pixel.")
    st.caption("Compter environ 30 s par été. Seule la fenêtre autour du site est lue dans les images.")
    if meta:
        st.caption(f"Données en local : étés {', '.join(map(str, meta['annees']))}, "
                   f"{meta['dates_retenues']} dates retenues sur {meta['scenes']} scènes "
                   f"(mise à jour {meta['maj'].replace('T', ' ')}).")
    if st.button("Télécharger" if not ready else "Mettre à jour", type="primary", disabled=not (years and months)):
        bar = st.progress(0.0, text="Landsat : recherche des scènes…")
        try:
            landsat.fetch(lat, lon, radius, years, months, max_cloud,
                          progress=lambda f: bar.progress(f, text=f"Landsat : lecture des scènes {f:.0%}"))
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Landsat : échec ({exc})")

if not ready:
    st.stop()

res = load(lat, lon, radius, (landsat.site_dir(lat, lon, radius) / "meta.json").stat().st_mtime)
scenes = res["scenes"]
if res["composite"] is None:
    st.warning("Aucune date exploitable (trop de nuages) : élargis les étés ou le seuil de nuages.")
    st.stop()

# --------------------------------------------------------------------------- #
# Chiffres clés
# --------------------------------------------------------------------------- #
s = landsat.summary(res)
k1, k2, k3, k4 = st.columns(4)
k1.metric("LST médiane au site", f"{s['site']:.1f} °C", help="Médiane des dates retenues, 3 × 3 pixels (90 m).")
k2.metric(f"Moyenne de la zone ({radius} m)", f"{s['zone']:.1f} °C")
k3.metric("Écart site – zone", f"{s['ecart_median']:+.1f} °C",
          help="Médiane, sur les dates retenues, de l'écart du jour entre le site et la moyenne de la zone.")
k4.metric("Rang du site dans la zone", f"{s['rang']:.0%}",
          help="Part des pixels de la zone plus froids que le site (100 % = point le plus chaud).")
st.caption(f"{s['dates']} dates retenues (au moins {landsat.MIN_VALID:.0%} de la zone sans nuage).")

# --------------------------------------------------------------------------- #
# Carte du composite
# --------------------------------------------------------------------------- #
st.subheader("Carte : température de surface médiane")
comp = res["composite"]
vmin, vmax = (float(v) for v in np.nanpercentile(comp, [2, 98]))
fig = go.Figure()
fig.add_trace(disk_trace(lat, lon, radius))
fig.add_trace(site_marker(site))
fig.add_trace(colorbar_trace(lat, lon, vmin, vmax, "LST (°C)"))
zoom = {500: 15, 1000: 14, 2000: 13, 3000: 12.4}[radius]
st.plotly_chart(layout_map(fig, lat, lon, zoom=zoom, height=560,
                           layers=[raster_layer(comp, res["transform"], res["crs"], vmin, vmax)]))
st.caption(f"Pixels de 30 m (la bande thermique est mesurée à 100 m puis rééchantillonnée). "
           f"Échelle : {vmin:.1f} à {vmax:.1f} °C (2e – 98e centiles).")

# --------------------------------------------------------------------------- #
# Série par date
# --------------------------------------------------------------------------- #
st.subheader("Série par date de passage")
kept = scenes[scenes["retenue"]]
long = kept.melt(id_vars=["date"], value_vars=["lst_site", "lst_zone"], var_name="série", value_name="LST (°C)")
long["série"] = long["série"].map({"lst_site": "Site", "lst_zone": f"Moyenne de la zone ({radius} m)"})
fig = px.line(long, x="date", y="LST (°C)", color="série", markers=True)
fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
fig.update_traces(connectgaps=False)
st.plotly_chart(fig)

fig = px.bar(kept, x="date", y="ecart", labels={"ecart": "écart site – zone (°C)"})
fig.add_hline(y=0, line_dash="dot")
fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig)

with st.expander("Tableau des scènes"):
    show = scenes.drop(columns="scenes").rename(columns={
        "heure_utc": "heure (UTC)", "nuages_scene": "nuages scène (%)", "part_valide": "zone sans nuage",
        "lst_site": "LST site (°C)", "lst_zone": "LST zone (°C)", "ecart": "écart (°C)"})
    st.dataframe(show.round(2), hide_index=True)
    st.download_button("Exporter en CSV", show.to_csv(index=False).encode("utf-8"),
                       file_name=f"landsat_{landsat.site_key(lat, lon, radius)}.csv", mime="text/csv")
