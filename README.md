# Hélio – exploration des données de chaleur

Application Streamlit pour explorer, croiser et modéliser les données publiques
utiles à la prévision du stress thermique, sur toute la France (outre-mer compris).

## Installation (une fois)

```bash
conda activate helio            # ou ton venv
pip install -r requirements.txt
```

## Lancement

```bash
cd helio
streamlit run Accueil.py
```

Le site s'ouvre dans le navigateur (http://localhost:8501).

## Utilisation

1. Choisis un **département** dans la barre latérale (métropole ou outre-mer).
2. Onglet **Stations** → « Données de … » → coche une période (commence par
   `latest-…`, la plus légère) → **Télécharger**.
3. Le fichier est converti en Parquet dans `data/processed/` : les visites
   suivantes sont instantanées.
4. Onglet **Urbain** → choisis un **site** (adresse ou coordonnées) → télécharge
   le territoire LCZ qui le contient, puis les bâtiments BDNB autour.
5. Onglet **Satellite** → même site → télécharge les étés Landsat (≈ 30 s par été).
6. Onglet **ERA5** → clé Copernicus CDS dans `~/.cdsapirc` (voir l'onglet) → période →
   séries, comparaison avec une station, export EPW d'une année complète.
7. Onglet **AROME / ARPEGE** → clés `METEOFRANCE_AROME_KEY` / `METEOFRANCE_ARPEGE_KEY` dans
   `~/.zshrc` → run, variables, horizon → prévision au site (chaque run est archivé).
8. Onglet **Modèle** → préparer les stations (ERA5, BDNB, Sentinel-2…) → construire le jeu →
   entraîner et valider → prédire au site. Long : aussi en terminal (`python -m core.model --help`).
   LightGBM sur Mac : `brew install libomp` (sinon repli automatique sur scikit-learn).
9. Onglet **Croisement** → une station (cible connue) ou un site → état des sources,
   fiche fixe, table horaire, « Enregistrer pour le modèle » (`data/processed/croisement/`).

## Structure

```
Accueil.py                  page d'accueil
pages/                      un fichier par onglet
core/config.py              chemins, identifiants des jeux de données
core/zones.py               départements métropole + outre-mer
core/ui.py                  sélecteurs partagés (zone, site étudié)
core/net.py                 téléchargement en flux, requêtes avec reprises
core/maps.py                cartes plotly partagées (site, rayon, image raster)
core/epw.py                 écriture de fichiers météo EPW (EnergyPlus, UWG)
core/features.py            croisement : fiche fixe + table horaire d'un site (entrée du modèle)
core/model.py               jeu multi-stations, entraînement, validation, prédiction (+ CLI)
core/sources/               un module par source (télécharger / convertir / lire)
data/raw, data/processed    stockage local (non versionné)
prototypes/                 études ponctuelles (ex. UWG Marseille), hors application
```

## Notes

- Heures : UTC pour la métropole (converties en heure de Paris à l'affichage),
  heure locale pour l'outre-mer.
- Qualité : l'option « valeurs validées uniquement » garde les codes Q = 0 ou 1.
