"""Chemins et constantes partagés par toute l'application."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for d in (RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Jeu « Données climatologiques de base – horaires » (Météo-France, data.gouv.fr)
DATASET_HORAIRE_ID = "6569b4473bedf2e7abad3b72"
DATAGOUV_API = "https://www.data.gouv.fr/api"

HTTP_TIMEOUT = 120
