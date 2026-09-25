"""Hélio – exploration des données de chaleur (page d'accueil).

Lancement :  streamlit run Accueil.py
"""
import streamlit as st

st.set_page_config(page_title="Hélio – données chaleur", page_icon="☀️", layout="wide")

from core import ui  # noqa: E402

zone = ui.zone_selector()

st.title("☀️ Hélio – exploration des données")
st.markdown(
    """
Outil interne pour explorer, croiser et modéliser les données publiques utiles à la
prévision du **stress thermique site par site**, sur **toute la France, outre-mer compris**.

Choisis une **zone** dans la barre latérale : chaque onglet télécharge les données de
cette zone à la demande, les garde en local (`data/`), puis les affiche.
"""
)

st.subheader("Onglets")
st.markdown(
    """
| Onglet | Sources | État |
|---|---|---|
| **Stations** | Météo-France, climatologie horaire (data.gouv) | ✅ disponible |
| **AROME / ARPEGE** | Portail API Météo-France (clé requise) | à venir |
| **ERA5** | Copernicus CDS (clé requise) | à venir |
| **Satellite** | Landsat (Planetary Computer), Sentinel (Copernicus) | à venir |
| **Urbain** | LCZ Cerema, BDNB | à venir |
| **Croisement** | Toutes les sources, en un point | à venir |
| **Modèle** | LightGBM, validation par station exclue | à venir |
"""
)

st.subheader("Couverture outre-mer, source par source")
st.markdown(
    """
- **Stations Météo-France, ERA5, Landsat, Sentinel** : métropole et outre-mer.
- **AROME** : domaines séparés pour l'outre-mer (Antilles, Guyane, océan Indien,
  Nouvelle-Calédonie, Polynésie), en 0,025°.
- **LCZ Cerema** : 93 territoires, dont Guadeloupe, Martinique, Guyane, La Réunion, Mayotte.
- **BDNB, Sat4BDNB** : à vérifier pour l'outre-mer (principalement la métropole).
"""
)
