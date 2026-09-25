import streamlit as st

st.set_page_config(page_title="Hélio – Croisement", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🔀 Croisement", ['Clic sur un point : extraction de toutes les variables disponibles.', "Construction de la table d'entrée du modèle (une ligne par site et par heure)."])
