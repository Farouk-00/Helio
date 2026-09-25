import streamlit as st

st.set_page_config(page_title="Hélio – Modèle", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🤖 Modèle", ['LightGBM : prédire la température au site.', 'Validation par station exclue + découpage temporel.', 'Comparaison aux références (maille brute, station de référence).'])
