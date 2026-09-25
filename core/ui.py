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
