import streamlit as st

st.set_page_config(page_title="Hélio – Urbain", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🏙️ Urbain", ['LCZ Cerema (classe + hauteur, part bâtie, végétation…).', 'BDNB : caractéristiques des bâtiments.', 'Sat4BDNB : intensité ICU et vulnérabilité (IRIS).'])
