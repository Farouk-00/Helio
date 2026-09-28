"""Éléments d'interface partagés entre les pages."""
from __future__ import annotations

import streamlit as st

from core import zones


def zone_selector(available: list[str] | None = None) -> str:
    """Sélecteur de département dans la barre latérale, mémorisé entre pages."""
    codes = available or list(zones.DEPARTEMENTS)
    codes = sorted(set(codes), key=lambda c: (len(c), c))
    current = st.session_state.get("zone", "13")
    if current not in codes:
        current = codes[0]
    with st.sidebar:
        st.markdown("### Zone d'étude")
        zone = st.selectbox(
            "Département (métropole ou outre-mer)",
            codes, index=codes.index(current), format_func=zones.label,
        )
    st.session_state["zone"] = zone
    if zones.is_outre_mer(zone):
        st.sidebar.info("Outre-mer : les heures des fichiers Météo-France sont en "
                        "heure locale, pas en UTC.")
    return zone


def coming_soon(title: str, points: list[str]) -> None:
    st.title(title)
    st.info("Onglet à venir.")
    st.markdown("\n".join(f"- {p}" for p in points))


@st.cache_data(ttl=3600, show_spinner=False)
def _geocode(query: str) -> list[dict]:
    from core.sources import geocodage
    return geocodage.search(query)


def site_selector() -> dict | None:
    """Choix du site étudié (adresse ou coordonnées), mémorisé entre pages.

    Renvoie {"lat", "lon", "label"} ou None.
    """
    site = st.session_state.get("site")
    with st.container(border=True):
        st.markdown("#### 📍 Site étudié")
        mode = st.radio("Localiser le site par", ["Adresse", "Coordonnées"], horizontal=True,
                        label_visibility="collapsed")
        if mode == "Adresse":
            query = st.text_input("Adresse", placeholder="ex. 10 La Canebière, Marseille")
            if query:
                try:
                    hits = _geocode(query)
                except Exception as exc:  # réseau indisponible
                    hits = []
                    st.error(f"Géocodage indisponible : {exc}")
                if hits:
                    c1, c2 = st.columns([4, 1])
                    i = c1.selectbox("Résultat", range(len(hits)), format_func=lambda k: hits[k]["label"])
                    c2.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
                    if c2.button("Choisir", type="primary"):
                        site = {k: hits[i][k] for k in ("lat", "lon", "label")}
                elif query.strip():
                    st.warning("Aucune adresse trouvée.")
        else:
            c1, c2, c3 = st.columns([2, 2, 1])
            lat = c1.number_input("Latitude", value=float(site["lat"]) if site else 43.29540,
                                  format="%.5f", step=0.0001)
            lon = c2.number_input("Longitude", value=float(site["lon"]) if site else 5.37525,
                                  format="%.5f", step=0.0001)
            c3.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
            if c3.button("Choisir", type="primary"):
                site = {"lat": lat, "lon": lon, "label": f"{lat:.5f}, {lon:.5f}"}
        if site:
            coords = f"{site['lat']:.5f}, {site['lon']:.5f}"
            extra = "" if site["label"] == coords else f" ({coords})"
            st.caption(f"Site actuel : **{site['label']}**{extra}")
    st.session_state["site"] = site
    return site
