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
DATAGOUV_RESOURCE = "https://www.data.gouv.fr/fr/datasets/r/{rid}"  # lien stable vers un fichier

# LCZ Cerema : liste des 93 territoires (contour + lien data.gouv) et tuiles nationales
CEREMA_ARCGIS = "https://cartagene.cerema.fr/server/rest/services/Hosted"
LCZ_TERRITOIRES_URL = f"{CEREMA_ARCGIS}/aires_urbaines_3857/FeatureServer/0/query"
LCZ_COMMUNES_URL = f"{CEREMA_ARCGIS}/statistiques_commune_3857/FeatureServer/0/query"
LCZ_TILES_URL = f"{CEREMA_ARCGIS}/l_lcz_spot_000_2022_tl/MapServer/tile/{{z}}/{{y}}/{{x}}"
LCZ_TILES_MAXZOOM = 14  # au-delà, le serveur renvoie 404

# BDNB : API ouverte (sans clé : 10 lignes par requête, 120 requêtes / minute)
BDNB_API = "https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet"

# Géocodage d'adresses (Géoplateforme IGN, remplace api-adresse.data.gouv.fr)
GEOCODAGE_URL = "https://data.geopf.fr/geocodage/search"

# Planetary Computer (Microsoft) : catalogue STAC, sans clé (URL signées automatiquement)
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Altitude et limite terre-mer (IGN, Géoplateforme) ; trait de côte Natural Earth 1:10 M en repli
ALTI_URL = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
IGN_WFS_URL = "https://data.geopf.fr/wfs/ows"
COASTLINE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
                 "geojson/ne_10m_coastline.geojson")

HTTP_TIMEOUT = 120
