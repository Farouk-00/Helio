"""Onglet Urbain : zones climatiques locales (Cerema) et bâtiments (BDNB) autour d'un site."""
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hélio – Urbain", page_icon="🏙️", layout="wide")

from core import ui, zones  # noqa: E402
from core.maps import disk_trace, layout_map, outline, site_marker  # noqa: E402
from core.sources import bdnb  # noqa: E402
from core.sources import cerema_lcz as lcz  # noqa: E402


# --------------------------------------------------------------------------- #
# Données (mises en cache)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Lecture de la liste des territoires LCZ…")
def catalog(refresh_token: int = 0) -> gpd.GeoDataFrame:
    return lcz.get_catalog(refresh=refresh_token > 0)


@st.cache_data(show_spinner="Rattachement des territoires au département…")
def territories(dep: str, refresh_token: int = 0) -> gpd.GeoDataFrame:
    return lcz.territories_for(dep, catalog(refresh_token))


@st.cache_resource(show_spinner="Chargement des LCZ du territoire…")
def load_lcz(code: str, mtime: float) -> gpd.GeoDataFrame:  # mtime : relit après re-téléchargement
    return lcz.load(code)


@st.cache_data(show_spinner=False)
def load_bdnb(lat: float, lon: float, radius: int, mtime: float) -> gpd.GeoDataFrame:
    return bdnb.load(lat, lon, radius)


# --------------------------------------------------------------------------- #
# Zone, catalogue, site
# --------------------------------------------------------------------------- #
zone = ui.zone_selector()
st.title("🏙️ Urbain")
st.caption("Sources : Cerema, « Zones climatiques locales (LCZ) » 2022 (data.gouv.fr) · "
           "CSTB, Base de données nationale des bâtiments (API ouverte BDNB).")

refresh = st.session_state.get("lcz_refresh", 0)
try:
    cat = catalog(refresh)
    zone_terr = territories(zone, refresh)
    cat_error = None
except Exception as exc:  # réseau indisponible, service modifié…
    cat = gpd.GeoDataFrame({"code": [], "nom": [], "url": []}, geometry=[], crs="EPSG:4326")
    zone_terr = cat.assign(n_communes=[])
    cat_error = exc
if cat_error is not None:
    st.error(f"Liste des territoires LCZ inaccessible (service Cerema) : {cat_error}")

site = ui.site_selector()
radius = st.sidebar.select_slider("Rayon autour du site (m)", options=[50, 100, 150, 200, 300, 500],
                                  value=st.session_state.get("rayon", 150),
                                  help="Pour la composition LCZ et les bâtiments BDNB. "
                                       "BDNB : environ 30 s pour 150 m en centre-ville.")
st.session_state["rayon"] = radius
names = dict(zip(cat["code"], cat["nom"]))

# --------------------------------------------------------------------------- #
# LCZ : téléchargement par territoire
# --------------------------------------------------------------------------- #
st.header("Zones climatiques locales (LCZ)")
done = lcz.downloaded()
site_terr = lcz.territory_at(site["lat"], site["lon"], cat) if site and not cat.empty else None

with st.expander(f"📥 Territoires LCZ de {zones.label(zone)}",
                 expanded=not set(zone_terr["code"]) & set(done)):
    if zone_terr.empty:
        st.warning("Aucun territoire LCZ ne couvre ce département : le Cerema ne cartographie que "
                   "les 93 aires urbaines de plus de 50 000 habitants.")
    else:
        show = zone_terr[["nom", "code", "n_communes"]].rename(
            columns={"n_communes": "communes du département"})
        show["téléchargé"] = show["code"].isin(done).map({True: "✅", False: ""})
        st.dataframe(pd.DataFrame(show), hide_index=True)
        default = [site_terr] if site_terr in set(zone_terr["code"]) - set(done) else []
        todo = st.multiselect("Territoires à télécharger", list(zone_terr["code"]), default=default,
                              format_func=lambda c: f"{names.get(c, c)} ({c})",
                              help="Zip de 20 à 230 Mo (shapefile + raster à 1,5 m). "
                                   "Seul le shapefile est gardé, en GeoParquet.")
        c1, c2 = st.columns([1, 3])
        if c1.button("Télécharger", type="primary", disabled=not todo):
            for code in todo:
                row = zone_terr[zone_terr["code"] == code].iloc[0]
                bar = st.progress(0.0, text=f"{row['nom']} : téléchargement…")
                try:
                    lcz.fetch(code, row["url"],
                              progress=lambda f, b=bar, n=row["nom"]: b.progress(f, text=f"{n} : {f:.0%}"))
                    bar.progress(1.0, text=f"{row['nom']} : converti en GeoParquet ✅")
                except Exception as exc:
                    st.error(f"{row['nom']} : échec ({exc})")
            st.cache_resource.clear()
            st.rerun()
        if c2.button("Actualiser la liste"):
            st.session_state["lcz_refresh"] = refresh + 1
            st.cache_data.clear()
            st.rerun()

# Carte d'ensemble : tuiles Cerema + contours des territoires du département
if not zone_terr.empty or site:
    fig = go.Figure()
    for r in zone_terr.itertuples():
        lons, lats = outline(r.geometry)
        fig.add_trace(go.Scattermap(lon=lons, lat=lats, mode="lines", line=dict(width=2, color="#1f4e8c"),
                                    name=r.nom, hoverinfo="name", showlegend=False))
    if site:
        fig.add_trace(site_marker(site))
        center, zoom = (site["lat"], site["lon"]), 11
    else:
        xmin, ymin, xmax, ymax = zone_terr.total_bounds
        center, zoom = ((ymin + ymax) / 2, (xmin + xmax) / 2), 8.5
    st.plotly_chart(layout_map(fig, *center, zoom=zoom, height=450, lcz_tiles=True))
    st.caption("Fond : LCZ du Cerema (tuiles nationales, visibles jusqu'au zoom 14 ; "
               "au-delà, voir la carte du site plus bas). Traits bleus : territoires couverts.")
    leg = " · ".join(f"<span style='color:{col}'>■</span> {code} {lib}"
                     for code, (lib, col) in lcz.LCZ_CLASSES.items() if code != "10")
    st.markdown(f"<small>{leg}</small>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Fiche du site
# --------------------------------------------------------------------------- #
st.header("Fiche du site")
if not site:
    st.info("Choisis un site ci-dessus (adresse ou coordonnées) pour afficher sa fiche.")
    st.stop()
lat, lon = site["lat"], site["lon"]

# --- LCZ au site ----------------------------------------------------------- #
st.subheader(f"LCZ dans un rayon de {radius} m")
prof = None
if site_terr is None:
    st.info("Le site n'est dans aucun des 93 territoires LCZ du Cerema.")
elif site_terr not in done:
    st.info(f"Le site est dans le territoire **{names.get(site_terr, site_terr)}** : "
            "télécharge-le dans la liste ci-dessus pour afficher sa fiche LCZ.")
else:
    gdf = load_lcz(site_terr, lcz.parquet_path(site_terr).stat().st_mtime)
    prof = lcz.site_profile(gdf, lat, lon, radius)
    if prof["composition"].empty:
        st.warning("Aucun polygone LCZ dans ce rayon.")
    else:
        comp = prof["composition"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Classe au site", f"LCZ {prof['classe']}" if prof["classe"] else "–",
                  help=lcz.label(prof["classe"]))
        k2.metric("Classe dominante dans le rayon", f"LCZ {comp.iloc[0]['lcz_code']}",
                  f"{comp.iloc[0]['part']:.0%} de la surface", delta_color="off",
                  help=comp.iloc[0]["libellé"])
        built = comp[comp["lcz_code"].str.isdigit()]["part"].sum()
        k3.metric("Part des classes bâties (1 à 10)", f"{built:.0%}")
        st.caption(f"Au site : {lcz.label(prof['classe'])}")

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(comp, x="part", y="libellé", orientation="h", color="lcz_code",
                         color_discrete_map={k: v[1] for k, v in lcz.LCZ_CLASSES.items()},
                         labels={"part": "part de la surface", "libellé": ""})
            fig.update_layout(height=60 + 32 * len(comp), showlegend=False,
                              margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%",
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig)
        with c2:
            if not prof["indicateurs"].empty:
                st.markdown("**Indicateurs moyens dans le rayon** (pondérés par la surface)")
                st.dataframe(prof["indicateurs"].round(2).rename("valeur"))
                st.caption("Sens des champs Cerema (hre, bur, ver…) déduit du guide, à confirmer.")
        if prof["polygone"] is not None:
            with st.expander("Attributs du polygone LCZ au site"):
                st.dataframe(prof["polygone"].astype(str).rename("valeur"))

# --- Bâtiments BDNB -------------------------------------------------------- #
st.subheader(f"Bâtiments (BDNB) dans un rayon de {radius} m")
bat = None
if zones.is_outre_mer(zone):
    st.info("L'API ouverte BDNB ne couvre que la métropole.")
elif not bdnb.is_ready(lat, lon, radius):
    st.caption("L'API ouverte renvoie 10 bâtiments par requête (120 requêtes / minute) : "
               "compter environ 30 s pour 150 m en centre-ville, 2 min pour 300 m.")
    if st.button(f"Télécharger les bâtiments ({radius} m)", type="primary"):
        pbar = st.progress(0.0, text="BDNB : téléchargement…")
        try:
            bdnb.fetch(lat, lon, radius, progress=lambda f: pbar.progress(f, text=f"BDNB : {f:.0%}"))
            st.rerun()
        except Exception as exc:
            st.error(f"BDNB : échec ({exc})")
else:
    bat = load_bdnb(lat, lon, radius, bdnb.site_path(lat, lon, radius).stat().st_mtime)
    if bat.empty:
        st.warning("Aucun bâtiment BDNB dans ce rayon.")
        bat = None
    else:
        s = bdnb.summary(bat, lat, lon, radius)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Bâtiments", s["n"])
        k2.metric("Part de sol bâti", f"{s['part_batie']:.0%}")
        k3.metric("Hauteur moyenne", f"{s['hauteur_moy']:.1f} m" if s["hauteur_moy"] else "–",
                  help="Moyenne pondérée par l'emprise au sol.")
        k4.metric("Année de construction médiane", f"{s['annee_med']:.0f}" if s["annee_med"] else "–")

        c1, c2 = st.columns(2)
        usage = bat["usage_principal_bdnb_open"].fillna("inconnu").value_counts().rename_axis("usage")
        fig = px.bar(usage.reset_index(name="bâtiments"), x="bâtiments", y="usage", orientation="h")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="",
                          yaxis=dict(categoryorder="total ascending"))
        c1.plotly_chart(fig)
        fig = px.histogram(bat, x="annee_construction", nbins=30,
                           labels={"annee_construction": "année de construction"})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="bâtiments")
        c2.plotly_chart(fig)

        with st.expander("Tableau des bâtiments"):
            table = pd.DataFrame(bat.drop(columns="geometry")).rename(
                columns={**bdnb.COLUMNS, "distance_m": "Distance au site (m)"})
            st.dataframe(table, hide_index=True)
            st.download_button("Exporter en CSV", table.to_csv(index=False).encode("utf-8"),
                               file_name=f"bdnb_{lat:.5f}_{lon:.5f}_r{radius}.csv", mime="text/csv")

# --- Carte du site --------------------------------------------------------- #
has_lcz = prof is not None and not prof["near"].empty
if has_lcz or bat is not None:
    st.subheader("Carte du site")
    fig = go.Figure()
    if has_lcz:
        near = prof["near"].to_crs("EPSG:4326")
        for code, part in near.groupby("lcz_code"):
            part = part.reset_index(drop=True)
            col = lcz.LCZ_CLASSES[code][1]
            fig.add_trace(go.Choroplethmap(
                geojson=part.geometry.__geo_interface__, locations=[str(i) for i in range(len(part))],
                z=[1] * len(part), colorscale=[[0, col], [1, col]], showscale=False,
                marker=dict(opacity=0.55, line=dict(width=0)), name=lcz.label(code), showlegend=True,
                hovertext=[lcz.label(code)] * len(part), hoverinfo="text"))
    if bat is not None:
        b = bat.to_crs("EPSG:4326").reset_index(drop=True)
        hover = (b["libelle_adr_principale_ban"].fillna("adresse inconnue") + "<br>"
                 + b["usage_principal_bdnb_open"].fillna("usage inconnu") + " · "
                 + b["hauteur_mean"].map(lambda h: f"{h:.0f} m" if pd.notna(h) else "hauteur inconnue")
                 + " · " + b["annee_construction"].map(lambda a: f"{a:.0f}" if pd.notna(a) else "année inconnue"))
        hmax = b["hauteur_mean"].max()
        fig.add_trace(go.Choroplethmap(
            geojson=b.geometry.__geo_interface__, locations=[str(i) for i in range(len(b))],
            z=b["hauteur_mean"], colorscale="Greys", zmin=0, zmax=max(float(hmax) if pd.notna(hmax) else 0, 10),
            marker=dict(opacity=0.85, line=dict(width=0.5, color="#333")), name="Bâtiments (hauteur)",
            colorbar=dict(title="hauteur (m)", thickness=12, len=0.5, x=0.99),
            hovertext=hover, hoverinfo="text"))
    fig.add_trace(disk_trace(lat, lon, radius))
    fig.add_trace(site_marker(site))
    zoom = {50: 18, 100: 17, 150: 16.5, 200: 16, 300: 15.5, 500: 14.8}[radius]
    st.plotly_chart(layout_map(fig, lat, lon, zoom=zoom, height=560))
    st.caption("Couleurs : classes LCZ (légende Cerema). Gris : bâtiments BDNB, du clair au foncé selon la hauteur.")
