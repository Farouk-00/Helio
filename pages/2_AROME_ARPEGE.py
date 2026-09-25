import streamlit as st

st.set_page_config(page_title="Hélio – AROME / ARPEGE", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🛰️ AROME / ARPEGE", ['Clé du portail API Météo-France (API ciblée Modèles).', 'Métropole : AROME 0,01° (J+2), ARPEGE 0,1° (J+4). Outre-mer : AROME-OM 0,025°.', "Carte d'une variable (T 2 m, HR 2 m, vent 10 m, rayonnement) pour une échéance.", 'Prévision en un point, comparée aux observations des stations.'])
