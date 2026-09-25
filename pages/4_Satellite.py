import streamlit as st

st.set_page_config(page_title="Hélio – Satellite", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🛰️ Satellite", ['Landsat 8/9 via Planetary Computer : température de surface estivale à 30 m.', 'Sentinel-2 : NDVI (végétation).', 'Sentinel-3 via openEO : température de surface nocturne à 1 km.'])
