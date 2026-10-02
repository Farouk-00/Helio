# Pipeline Hélio : données → jeu d'entraînement → modèle PyTorch

Ce dossier est indépendant du site Streamlit : aucun import de `core/`, pas d'interface graphique.
Chaque fichier se lit seul et se lance seul.

## Installation (une fois)

```bash
cd pipeline
python3 -m pip install -r requirements.txt      # PyTorch : ~300 Mo
```

Clés (déjà en place si le site fonctionne) : `~/.cdsapirc` pour ERA5,
`METEOFRANCE_AROME_KEY` / `METEOFRANCE_ARPEGE_KEY` dans `~/.zshrc` pour AROME / ARPEGE.
Les autres sources sont sans clé.

## Les fichiers

| fichier | rôle |
|---|---|
| `config.py` | départements, période, rayons, dossier des données (`pipeline/donnees/`, non versionné) |
| `outils.py` | requêtes HTTP avec reprises, position du soleil, humidité |
| `api_meteofrance_stations.py` | mesures horaires des stations (data.gouv) : la **cible** |
| `api_era5.py` | réanalyse ERA5-Land au point (Copernicus CDS) : la **météo de fond** |
| `api_ign.py` | altitude, creux / hauteur (TPI), écart avec la maille ERA5, distance à la mer |
| `api_cerema_lcz.py` | zones climatiques locales (classe au site, composition, indicateurs) |
| `api_bdnb.py` | bâtiments : hauteur, emprise, façades, année, usages |
| `api_landsat.py` | température de surface d'été (site vs alentours) |
| `api_sentinel2.py` | végétation (NDVI, parts d'arbres et d'herbe) |
| `api_arome_arpege.py` | prévisions Météo-France au site (pour la prévision réelle, plus tard) |
| `construire_jeu.py` | **étape 1** : assemble tout en une table station × heure par département |
| `modele.py` | **étape 2** : split train / valid / test, modèle GRU ou MLP, entraînement, scores |

Chaque `api_*.py` a la même forme : `telecharger(...)` (avec cache) et/ou `caracteristiques(lat, lon, ...)`
qui renvoie un dictionnaire de variables. Exemple : `python api_ign.py 43.3048 5.3955`.

## Étape 1 : construire le jeu

```bash
python construire_jeu.py                          # départements et période de config.py
python construire_jeu.py --dep 84 --sans landsat  # un département, sans une source
```

Sortie : `donnees/jeu/jeu_<dep>.parquet`, une ligne par station et par heure, heures continues
(T_station = NaN quand la mesure manque). Tout est en cache : relancer ne retélécharge rien.
Durée : surtout ERA5 (~1 min par station) et Landsat (~30 s par station).

**Réutiliser ce que le site a déjà téléchargé** (évite de retélécharger ERA5) :

```bash
mkdir -p donnees/era5 donnees/stations donnees/bdnb donnees/lcz donnees/sentinel2
cp ../data/processed/era5/*.parquet donnees/era5/
cp ../data/processed/stations_horaires/*.parquet donnees/stations/
cp ../data/processed/bdnb/*.parquet donnees/bdnb/
cp ../data/processed/lcz/*.parquet donnees/lcz/
cp ../data/raw/lcz/_territoires.geojson donnees/lcz/
```

## Étape 2 : entraîner

```bash
python modele.py
```

Tous les réglages sont dans le dictionnaire `CONFIG` en haut de `modele.py` : variables,
découpage, architecture, entraînement. Pour régler les hyperparamètres depuis un notebook :

```python
import modele
df = modele.charger(["13", "84"])                 # chargé une fois
for lr in (1e-3, 3e-4):
    r = modele.entrainer({**modele.CONFIG, "departements": ["13", "84"], "lr": lr}, df=df, sauver=False)
    print(lr, modele.tableau_rmse(r["scores_valid"]))
```

### Ce que fait le modèle

- Cible : `T_station − era5_T` (la correction locale). Prévision = `era5_T + correction`.
- Entrée 1 : les `fenetre` dernières heures d'ERA5 au point (T, humidité, vent, rayonnement,
  infrarouge, pluie, soleil, heure), lues par un **GRU** (ou aplaties pour un **MLP**).
- Entrée 2 : les variables fixes du site (relief, mer, LCZ, bâti, satellite), normalisées ;
  une source absente vaut 0 avec un indicateur « manquant ».
- Perte MSE (ou Huber), AdamW, taux d'apprentissage divisé par 2 quand valid stagne,
  arrêt anticipé sur la RMSE valid, meilleur modèle sauvé dans `donnees/modeles/<nom>.pt`.

### Le découpage

Les stations sont tirées au hasard (graine fixe) en trois groupes :
**train** (apprentissage), **valid** (arrêt anticipé, choix des hyperparamètres),
**test** (score final, à ne regarder qu'à la fin). Avec `date_test`, train et valid sont
avant la date et test après : stations jamais vues **et** période future, la situation
d'un nouveau client. Pour juger la canicule, placer `date_test` avant un été.

### Scores affichés

RMSE (°C) de trois prévisions : ERA5 brut, ERA5 + biais moyen du train, modèle ; sur toutes
les heures, la nuit (21 h - 6 h), les heures ≥ 30 °C, et les Tmin / Tmax journalières.
Référence (LightGBM du site, 13, stations exclues + période future) : 1,88 → 1,73 °C ;
Tmin 2,14 → 2,03 °C.

Avec un seul département (~19 stations), valid ne compte que 3 stations : scores bruités et
surapprentissage rapide. Combiner 4-5 départements (~80 stations) change la donne.
