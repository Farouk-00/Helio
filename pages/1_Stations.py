"""Onglet Stations : observations horaires Météo-France, par département."""
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Hélio – Stations", page_icon="🌡️", layout="wide")

from core import ui, zones  # noqa: E402
from core.sources import meteofrance_stations as mfs  # noqa: E402


# --------------------------------------------------------------------------- #
# Catalogue et zone
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Lecture du catalogue data.gouv…")
def catalog(refresh_token: int = 0) -> pd.DataFrame:
    return mfs.get_catalog(refresh=refresh_token > 0)


try:
    cat = catalog(st.session_state.get("cat_refresh", 0))
    cat_error = None
except Exception as exc:  # réseau indisponible, API modifiée…
    cat, cat_error = pd.DataFrame(columns=["departement", "periode", "fichier", "url", "taille_mo"]), exc

available = sorted(set(cat["departement"]) | {f.stem.split("_")[1] for f in mfs.PROC.glob("H_*.parquet")})
zone = ui.zone_selector(available or None)

st.title("🌡️ Stations Météo-France")
st.caption("Source : Météo-France, « Données climatologiques de base – horaires » (data.gouv.fr). "
           "Une ligne par station et par heure, mesures validées.")

if cat_error is not None:
    st.error(f"Catalogue data.gouv inaccessible : {cat_error}")

# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #
zone_cat = cat[cat["departement"] == zone]
done = mfs.downloaded_periods(zone)

with st.expander(f"📥 Données de {zones.label(zone)}", expanded=not done):
    if zone_cat.empty:
        st.warning("Aucun fichier horaire trouvé pour cette zone dans le catalogue.")
    else:
        show = zone_cat[["periode", "fichier", "taille_mo"]].copy()
        show["téléchargé"] = show["periode"].isin(done).map({True: "✅", False: ""})
        st.dataframe(show, hide_index=True)
        todo = st.multiselect(
            "Périodes à télécharger", list(zone_cat["periode"]),
            default=[p for p in zone_cat["periode"] if "latest" in p and p not in done],
            help="Commence par la période la plus récente (« latest »), plus légère.",
        )
        c1, c2 = st.columns([1, 3])
        if c1.button("Télécharger", type="primary", disabled=not todo):
            for per in todo:
                row = zone_cat[zone_cat["periode"] == per].iloc[0]
                bar = st.progress(0.0, text=f"{row['fichier']} : téléchargement…")
                try:
                    mfs.fetch(zone, per, row["url"],
                              progress=lambda f, b=bar, n=row["fichier"]: b.progress(f, text=f"{n} : {f:.0%}"))
                    bar.progress(1.0, text=f"{row['fichier']} : converti en Parquet ✅")
                except Exception as exc:
                    st.error(f"{row['fichier']} : échec ({exc})")
            st.cache_data.clear()
            st.rerun()
        if c2.button("Actualiser le catalogue"):
            st.session_state["cat_refresh"] = st.session_state.get("cat_refresh", 0) + 1
            st.cache_data.clear()
            st.rerun()

if not done:
    st.info("Aucune donnée téléchargée pour cette zone : choisis une période ci-dessus.")
    st.stop()


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Chargement des observations…")
def load_zone(dep: str, periods: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = mfs.load(dep, list(periods))
    return df, mfs.stations_summary(df)


periods = tuple(st.sidebar.multiselect("Périodes chargées", done, default=done[-1:]) or done[-1:])
df, stations = load_zone(zone, periods)
if df.empty:
    st.stop()

outre_mer = zones.is_outre_mer(zone)
if not outre_mer:  # métropole : fichiers en UTC -> heure de Paris pour l'affichage
    df = df.assign(heure_locale=df["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.tz_localize(None))
else:
    df = df.assign(heure_locale=df["time"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stations", len(stations))
k2.metric("Avec température", int((stations.get("part_T", 0) > 0.5).sum()))
k3.metric("Avec rayonnement", int((stations.get("part_GLO", 0) > 0.5).sum()))
k4.metric("Période", f"{df['time'].min():%m/%y} → {df['time'].max():%m/%y}")

# --------------------------------------------------------------------------- #
# Carte
# --------------------------------------------------------------------------- #
st.subheader("Carte des stations")
m = stations.dropna(subset=["lat", "lon"]).copy()
m["couleur"] = m["part_T"].apply(lambda p: [214, 69, 65, 200] if p > 0.5 else [140, 140, 140, 160]) \
    if "part_T" in m else [[140, 140, 140, 160]] * len(m)
m["debut_txt"] = m["debut"].dt.strftime("%Y-%m-%d")
m["fin_txt"] = m["fin"].dt.strftime("%Y-%m-%d")
st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer("ScatterplotLayer", data=m, get_position="[lon, lat]",
                      get_fill_color="couleur", get_radius=1200, radius_min_pixels=4,
                      pickable=True)],
    initial_view_state=pdk.ViewState(latitude=float(m["lat"].mean()),
                                     longitude=float(m["lon"].mean()), zoom=7.5),
    tooltip={"text": "{nom} ({NUM_POSTE})\nAltitude : {alti} m\n{debut_txt} → {fin_txt}"},
))
st.caption("Rouge : température disponible sur plus de la moitié des heures. Gris : autres stations.")

with st.expander("Tableau des stations"):
    st.dataframe(stations, hide_index=True)

# --------------------------------------------------------------------------- #
# Série d'une station
# --------------------------------------------------------------------------- #
st.subheader("Série temporelle d'une station")
labels = {r.NUM_POSTE: f"{r.nom} ({r.NUM_POSTE}, {r.alti:.0f} m)" for r in stations.itertuples()}
order = stations.sort_values("n_heures", ascending=False)["NUM_POSTE"].tolist()
c1, c2, c3 = st.columns([2, 1, 1])
poste = c1.selectbox("Station", order, format_func=labels.get)
vars_dispo = [v for v in mfs.MEASURES if v in df.columns and df.loc[df["NUM_POSTE"] == poste, v].notna().any()]
var = c2.selectbox("Variable", vars_dispo, format_func=lambda v: f"{v} – {mfs.MEASURES[v]}") if vars_dispo else None
qual = c3.checkbox("Valeurs validées uniquement", value=True)

if var:
    s = df[df["NUM_POSTE"] == poste]
    tmin, tmax = s["heure_locale"].min().date(), s["heure_locale"].max().date()
    debut, fin = st.slider("Période affichée", min_value=tmin, max_value=tmax,
                           value=(max(tmin, tmax - pd.Timedelta(days=30)), tmax), format="DD/MM/YYYY")
    s = s[(s["heure_locale"].dt.date >= debut) & (s["heure_locale"].dt.date <= fin)]
    y = mfs.filter_quality(s, var) if qual else s[var]
    fig = px.line(x=s["heure_locale"], y=y, labels={"x": "Heure locale", "y": mfs.MEASURES[var]})
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Îlot de chaleur : écart à une station de référence
# --------------------------------------------------------------------------- #
st.subheader("Îlot de chaleur : écart de température à une station de référence")
st.caption("Choisis une station « de référence » (idéalement en périphérie, par exemple un aéroport) : "
           "l'écart de chaque station à cette référence approxime l'effet d'îlot de chaleur.")

with_t = stations[stations.get("part_T", 0) > 0.5]["NUM_POSTE"].tolist()
if len(with_t) < 2:
    st.info("Il faut au moins deux stations avec température pour cette analyse.")
    st.stop()

c1, c2 = st.columns(2)
REF_HINTS = ("AEROPORT", "AERO", "BLAGNAC", "MARIGNANE", "MERIGNAC", "BRON", "ORLY", "ROISSY",
             "FREJORGUES", "NICE", "RAIZET", "LAMENTIN", "GILLOT", "ROCHAMBEAU")
by_hours = stations.set_index("NUM_POSTE")["n_heures"]
with_t = sorted(with_t, key=lambda p: -by_hours[p])
ref_default = next((p for p in with_t if any(k in labels[p].upper() for k in REF_HINTS)), with_t[0])
ref = c1.selectbox("Station de référence", with_t, index=with_t.index(ref_default), format_func=labels.get)
mois = c2.multiselect("Mois étudiés", list(range(1, 13)), default=[6, 7, 8],
                      format_func=lambda i: ["janv", "févr", "mars", "avr", "mai", "juin", "juil",
                                             "août", "sept", "oct", "nov", "déc"][i - 1])

t = df[df["NUM_POSTE"].isin(with_t)].assign(Tq=lambda d: mfs.filter_quality(d, "T"))
t = t[t["heure_locale"].dt.month.isin(mois)]
wide = t.pivot_table(index="heure_locale", columns="NUM_POSTE", values="Tq")
if ref not in wide:
    st.warning("Pas de température validée pour la référence sur ces mois.")
    st.stop()
ecart = wide.sub(wide[ref], axis=0).drop(columns=ref)

# Profil moyen de l'écart selon l'heure de la journée
prof = ecart.groupby(ecart.index.hour).mean()
top = prof.mean().sort_values(ascending=False).head(8).index
prof_long = prof[top].rename(columns=labels).reset_index(names="heure").melt(
    id_vars="heure", var_name="station", value_name="écart (°C)")
fig = px.line(prof_long, x="heure", y="écart (°C)", color="station", markers=True)
fig.add_hline(y=0, line_dash="dot")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                  xaxis=dict(dtick=2, title="Heure locale"))
st.plotly_chart(fig)
st.caption("Profil moyen sur les mois choisis, pour les 8 stations en moyenne les plus chaudes que la référence. "
           "Un îlot de chaleur se voit typiquement par un écart plus fort la nuit.")

# Classement nocturne
nuit = ecart[(ecart.index.hour >= 22) | (ecart.index.hour <= 5)]
rank = (nuit.mean().rename("écart nocturne moyen (°C)").to_frame()
        .join(nuit.count().rename("heures"))
        .query("heures > 100").sort_values("écart nocturne moyen (°C)", ascending=False))
rank.index = rank.index.map(labels)
st.markdown("**Classement des stations par écart nocturne moyen (22 h – 5 h)**")
st.dataframe(rank.round(2))
