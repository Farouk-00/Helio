"""Réglages communs : où sont rangées les données, quels départements, quelle période.

Tout ce dossier est indépendant du site Streamlit (il n'importe rien de `core/`).
Les données téléchargées vont dans `pipeline/donnees/` (non versionné).
"""
from pathlib import Path

ICI = Path(__file__).resolve().parent
DONNEES = ICI / "donnees"            # un sous-dossier par source, créé à la demande

# Jeu d'entraînement
DEPARTEMENTS = ["13", "30", "83", "84", "04"]
DEBUT, FIN = "2025-01-01", "2025-12-31"   # période ERA5 + mesures (AAAA-MM-JJ, UTC)
RAYON_URBAIN_M = 150                      # BDNB, LCZ, Sentinel-2 : disque autour de la station
RAYON_SATELLITE_M = 1000                  # Landsat : disque pour l'écart site − alentours
ETE_SATELLITE = 2025                      # été (juin-août) utilisé pour Landsat et Sentinel-2
ERA5_JEU = "era5-land"                    # "era5-land" (0,1°) ou "era5" (0,25°, + nébulosité)

# Décalage heure locale normale - UTC (sans heure d'été). Les fichiers stations
# d'outre-mer sont en heure locale : ils sont ramenés en UTC avec ce décalage.
DECALAGE_UTC = {"971": -4, "972": -4, "973": -3, "974": 4, "975": -3, "976": 3,
                "977": -4, "978": -4, "986": 12, "987": -10, "988": 11}


def decalage_utc(departement: str) -> int:
    return DECALAGE_UTC.get(departement, 1)   # métropole : UTC+1 (heure d'hiver)


def dossier(nom: str) -> Path:
    """Sous-dossier de données d'une source (créé s'il n'existe pas)."""
    d = DONNEES / nom
    d.mkdir(parents=True, exist_ok=True)
    return d
