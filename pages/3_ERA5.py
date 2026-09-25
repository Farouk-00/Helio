import streamlit as st

st.set_page_config(page_title="Hélio – ERA5 / ERA5-Land", layout="wide")
from core import ui  # noqa: E402

ui.zone_selector()
ui.coming_soon("🌍 ERA5 / ERA5-Land", ['Clé Copernicus CDS (fichier ~/.cdsapirc).', 'ERA5-Land 0,1° : séries longues, rayonnement solaire.', 'ERA5-HEAT : UTCI de référence.'])
