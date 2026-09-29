"""Onglet ERA5 : réanalyse horaire ERA5 / ERA5-Land au site (Copernicus CDS)."""
import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Hélio – ERA5 / ERA5-Land", page_icon="🌍", layout="wide")

from core import epw, ui, zones  # noqa: E402
from core.sources import era5  # noqa: E402
from core.sources import meteofrance_stations as mfs  # noqa: E402


@st.cache_data(show_spinner="Chargement ERA5…")
def load(dataset: str, lat: float, lon: float, mtime: float) -> pd.DataFrame:  # mtime : relit après ajout
    return era5.load(dataset, lat, lon)


@st.cache_data(show_spinner="Chargement des observations…")
def load_stations(dep: str, periods: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = mfs.load(dep, list(periods))
    return df, mfs.stations_summary(df)


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


zone = ui.zone_selector()
st.title("🌍 ERA5 / ERA5-Land")
st.caption("Source : Copernicus Climate Change Service (C3S), réanalyses ERA5 et ERA5-Land, jeux « time-series » "
           "(point de grille le plus proche du site), licence CC-BY 4.0. Heures en **UTC**.")

site = ui.site_selector()
if not site:
    st.info("Choisis un site ci-dessus (adresse ou coordonnées).")
    st.stop()
lat, lon = site["lat"], site["lon"]

ds = st.radio("Jeu de données", list(era5.DATASETS), format_func=lambda k: era5.DATASETS[k]["libelle"])
glat, glon = era5.grid_point(ds, lat, lon)

if not era5.has_key():
    st.warning(
        "**Clé Copernicus CDS absente.** Pour l'obtenir (gratuit) :\n"
        "1. crée un compte sur [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) ;\n"
        "2. accepte la licence CC-BY sur la page du jeu (onglet *Download*, en bas) ;\n"
        "3. copie ton jeton (page de ton profil) dans le fichier `~/.cdsapirc` :\n"
        "```\nurl: https://cds.climate.copernicus.eu/api\nkey: <ton-jeton>\n```\n"
        "puis relance l'application.")

# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #
ready = era5.is_ready(ds, lat, lon)
today = dt.date.today()
end_max = today - dt.timedelta(days=era5.LATENCE_JOURS)
last_summer = today.year if today.month >= 9 else today.year - 1
with st.expander("📥 Téléchargement", expanded=not ready):
    period = st.date_input("Période", value=(dt.date(last_summer, 6, 1), dt.date(last_summer, 8, 31)),
                           min_value=dt.date(1950, 1, 2), max_value=end_max, format="DD/MM/YYYY")
    st.caption(f"Point de grille demandé : {glat:.2f} N, {glon:.2f} E. Les données arrivent avec environ "
               f"{era5.LATENCE_JOURS} jours de retard. Une année civile complète (non bissextile) permet "
               "d'exporter un fichier météo EPW, par exemple pour UWG.")
    valid = isinstance(period, tuple) and len(period) == 2
    if st.button("Télécharger", type="primary", disabled=not (valid and era5.has_key())):
        with st.spinner("Requête Copernicus en cours (de quelques secondes à quelques minutes)…"):
            try:
                era5.fetch(ds, lat, lon, period[0].isoformat(), period[1].isoformat())
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"ERA5 : échec ({exc})")

if not ready:
    st.stop()

df = load(ds, lat, lon, era5.site_path(ds, lat, lon).stat().st_mtime)
info = era5.grid_info(ds, lat, lon)
plat, plon = info.get("lat_grille", glat), info.get("lon_grille", glon)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Période en local", f"{df.index.min():%d/%m/%Y} → {df.index.max():%d/%m/%Y}")
k2.metric("Point de grille", f"{plat:.2f} N, {plon:.2f} E",
          f"à {distance_km(lat, lon, plat, plon):.1f} km du site", delta_color="off")
k3.metric("Température moyenne", f"{df['T'].mean():.1f} °C")
k4.metric("Température maximale", f"{df['T'].max():.1f} °C", f"le {df['T'].idxmax():%d/%m/%Y %H} h UTC",
          delta_color="off")

# --------------------------------------------------------------------------- #
# Séries
# --------------------------------------------------------------------------- #
st.subheader("Séries horaires")
tmin, tmax = df.index.min().date(), df.index.max().date()
debut, fin = st.slider("Période affichée", min_value=tmin, max_value=tmax,
                       value=(max(tmin, tmax - dt.timedelta(days=30)), tmax), format="DD/MM/YYYY")
view = df.loc[str(debut):f"{fin} 23:00"]
groups = {"Température": ["T", "TD", "TS"], "Rayonnement": ["GHI", "IR", "DNI", "DHI"],
          "Humidité": ["RH"], "Vent et pluie": ["WS", "RR"], "Nébulosité": ["N"]}
tabs = st.tabs([g for g, cols in groups.items() if any(c in view for c in cols)])
for tab, (g, cols) in zip(tabs, [(g, c) for g, c in groups.items() if any(x in view for x in c)]):
    cols = [c for c in cols if c in view]
    long = view[cols].rename(columns=era5.UNITES).reset_index().melt(id_vars="time", var_name="variable")
    fig = px.line(long, x="time", y="value", color="variable", labels={"time": "UTC", "value": ""})
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
    tab.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Comparaison avec une station Météo-France
# --------------------------------------------------------------------------- #
st.subheader("Comparaison avec une station Météo-France")
done = mfs.downloaded_periods(zone)
if zones.is_outre_mer(zone):
    st.info("Comparaison disponible en métropole seulement (heures locales dans les fichiers outre-mer).")
elif not done:
    st.info("Télécharge d'abord des données dans l'onglet Stations pour cette zone.")
else:
    obs, stations = load_stations(zone, tuple(done))
    stations = stations[stations.get("part_T", 0) > 0.5].copy()
    stations["km"] = distance_km(lat, lon, stations["lat"], stations["lon"])
    stations = stations.sort_values("km")
    poste = st.selectbox("Station", stations["NUM_POSTE"],
                         format_func=lambda p: "{} ({:.1f} km du site, {:.0f} m)".format(
                             *stations.set_index("NUM_POSTE").loc[p, ["nom", "km", "alti"]]))
    s = obs[obs["NUM_POSTE"] == poste].set_index("time")
    t_obs = mfs.filter_quality(s, "T")
    both = pd.DataFrame({"station": t_obs, "era5": df["T"]}).dropna()
    if len(both) < 24:
        st.info("Pas assez d'heures communes entre la station et ERA5 : télécharge la même période.")
    else:
        e = both["era5"] - both["station"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Heures communes", f"{len(both):,}".replace(",", " "))
        c2.metric("Biais ERA5 − station", f"{e.mean():+.2f} °C")
        c3.metric("Erreur absolue moyenne", f"{e.abs().mean():.2f} °C")
        c4.metric("Corrélation", f"{both.corr().iloc[0, 1]:.3f}")
        prof = e.groupby(e.index.hour).mean().rename("biais (°C)").rename_axis("heure UTC").reset_index()
        fig = px.bar(prof, x="heure UTC", y="biais (°C)")
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(dtick=2))
        st.plotly_chart(fig)
        st.caption("Biais moyen selon l'heure. ERA5 représente une maille de 9 à 28 km : l'écart avec la "
                   "station mêle erreur de la réanalyse et effets locaux (ville, relief, mer). "
                   "Une station en ville plus chaude que la maille la nuit, c'est l'îlot de chaleur.")

# --------------------------------------------------------------------------- #
# Export EPW
# --------------------------------------------------------------------------- #
st.subheader("Export d'un fichier météo EPW")
counts = df.groupby(df.index.year).size()
years = [int(y) for y, n in counts.items()
         if n >= 0.99 * 8760 and pd.Timestamp(year=int(y), month=12, day=31).dayofyear == 365]
if not years:
    st.caption("Télécharge une année civile complète non bissextile (ex. 2025) pour exporter un EPW.")
else:
    y = st.selectbox("Année", years[::-1])
    tz = zones.utc_offset(zone)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "era5.epw"
        filled = epw.write(df, path, year=y, tz_hours=tz, lat=plat, lon=plon, alt_m=0.0,
                           city=f"{ds.upper()} {plat:.2f}N {plon:.2f}E", source=f"Copernicus {ds}")
        data = path.read_bytes()
    st.download_button(f"Télécharger l'EPW {y}", data, file_name=f"{ds}_{plat:.2f}_{plon:.2f}_{y}.epw",
                       mime="text/plain")
    manq = {k: f"{v:.0%}" for k, v in filled.items() if v > 0}
    st.caption(f"Heure locale normale UTC{tz:+g}. Altitude inconnue (0 m). "
               + (f"Valeurs comblées par interpolation : {manq}. " if manq else "")
               + ("Rayonnement direct / diffus : ERA5 direct." if "DNI" in df else
                  "Rayonnement direct / diffus : décomposition d'Erbs (ERA5-Land n'a pas le direct)."))
