# CLAUDE.md – Projet Hélio

## Qui / comment travailler
- Porteur : Foucault. **Très bon niveau Python et ML**. Pas besoin d'expliquer les bases.
- Répondre **en français**, **court et direct**. Pas de longs pavés.
- **Ne rien modifier dans Notion** sans demande explicite (lecture seule par défaut).
- Machine : MacBook (Apple Silicon), zsh.
  - ⚠️ Le Python système macOS est en **3.9** (LibreSSL) : ne pas l'utiliser.
  - ⚠️ Le `conda` de `base` a un solveur libmamba cassé (`libarchive.20.dylib`) : éviter conda,
    ou utiliser `--solver=classic`.
  - Environnement recommandé : `/opt/anaconda3/bin/python3 -m venv ~/venvs/helio`
    puis `source ~/venvs/helio/bin/activate`.
- Garder le code **compatible Python 3.9** tant que l'env n'est pas stabilisé
  (`from __future__ import annotations`, pas de `match`, pas de `X | Y` hors annotations).

## Le projet
**Hélio** : startup qui prédit le **stress thermique site par site** (chantiers, entrepôts,
bâtiments), J+1/J+2 précis, J+3/J+4 en tendance, pour que les employeurs respectent le
**décret n° 2025-482 du 27 mai 2025** (obligations dès la vigilance canicule jaune).
Cibles : BTP, logistique, industrie, puis foncières, puis assureurs (vente de scores).
Moat visé : boucle de données propriétaires (capteurs clients) + archive de prévisions.

## Ce qu'on construit maintenant
Application **Streamlit** interne d'exploration, **toute la France, outre-mer compris**,
fonctionnant **zone par zone** (sélecteur de département dans la barre latérale).

Architecture en deux couches :
1. `core/sources/<source>.py` : télécharger → convertir (Parquet/NetCDF) → lire.
2. `pages/*.py` : n'affichent **que** le stockage local `data/processed/`.
   Chaque onglet a un bouton « télécharger / mettre à jour ».

Lancement : `python -m streamlit run Accueil.py`

### État des onglets
| Onglet | État | Sources |
|---|---|---|
| Stations | ✅ codé, testé sur données fictives seulement | Météo-France climatologie horaire (data.gouv) |
| Urbain | prochain | LCZ Cerema, BDNB (sans clé) |
| Satellite | à faire | Landsat via Planetary Computer, Sentinel-2/3 via openEO |
| AROME / ARPEGE | à faire | API ciblée Modèles Météo-France (clé requise) |
| ERA5 | à faire | Copernicus CDS (`~/.cdsapirc`) |
| Croisement | à faire | clic sur un point → toutes les variables → table d'entrée modèle |
| Modèle | à faire | LightGBM, validation par station exclue + split temporel |

### À vérifier au premier vrai run (non testable jusqu'ici)
- Catalogue data.gouv du jeu horaire (id `6569b4473bedf2e7abad3b72`) : noms de fichiers
  supposés `H_<dep>_<periode>.csv.gz` (regex dans `meteofrance_stations.py`).
- Format numérique des CSV (séparateur `;`, décimale supposée `.`, repli `,` géré).

## Décisions prises
- Langage : Python + Streamlit. Stockage local Parquet/NetCDF, déploiement plus tard sur
  Google Cloud (Cloud Run + Cloud Storage + Cloud Scheduler, région europe-west9).
- **Archivage des prévisions mis de côté pour l'instant** (mais c'est la seule donnée
  irremplaçable : AROME/ARPEGE ne sont conservés nulle part au-delà de 5 j (API) /
  14 j (fichiers)). Quand on s'y met : découper la zone (API ciblée avec lat/long),
  variables utiles seulement ; France entière en 0,025° ≈ 100 Go/an.
  Le miroir AWS `mf-nwp-models` a été vérifié : **vide** (fichiers statiques 2019).
- Pas de deep learning au MVP. LightGBM d'abord.

## Mémo des sources (faits vérifiés)
### Météo-France
- **Climatologie horaire** (meteo.data.gouv.fr / data.gouv) : **stations ponctuelles**
  (pas de maille), CSV.gz par département × période, **pas horaire**, heures **UTC en
  métropole**, **heure locale en outre-mer**. Colonnes : `NUM_POSTE, NOM_USUEL, LAT, LON,
  ALTI, AAAAMMJJHH`, puis mesures (`T, TD, TN, TX, U, FF, DD, FXI, RR1, GLO, INS, N, PMER…`),
  chacune avec `Q<X>` (0 protégée, 1 validée, 2 douteuse, 9 filtrée). `GLO` en J/cm², rare.
  Corse = « 20 », outre-mer sur 3 chiffres (971…988).
- **DPObs / Paquet Observations** : temps réel, **rétention 24 h**, horaire ou 6 min,
  CSV/JSON/GeoJSON. Utile seulement en production.
- **DPClim** (API Données Climatologiques) : **tout l'historique** d'une station (jusqu'à
  1850), asynchrone, **1 an max par requête**, dont **6 min** historique. CSV.
- **AROME** : ~1,3 km natif ; diffusé en 0,01° et 0,025° ; horaire jusqu'à **51 h**
  (runs 00/03/12/15 UTC ; 09/21 plus courts). Outre-mer : AROME-OM 0,025° par domaine.
  **SP1** = T/HR 2 m, vent 10 m (+rafales), pluie, nébulosité, **FLSOLAIRE_D (rayonnement)**.
  SP2/SP3 = additionnels (probablement TD 2 m, **WETBT 2 m**, nébulosité par étage,
  T surface, CAPE…) – à confirmer avec `grib_ls`. **HP1 France = vent + HR 10-100 m,
  pas de température.**
- **ARPEGE** : 0,1° Europe (~11×8 km à 43°N), jusqu'à **102 h**, ~4 runs/j.
- API ciblée Modèles (WCS) : 1 param × 1 niveau × 1 échéance par requête, subset
  `lat()/long()/time()/height()`, GRIB2 ou GeoTIFF, 50 req/min, rétention 5 j.
  Services : `MF-NWP-HIGHRES-AROME-001-FRANCE-WCS`, `MF-NWP-GLOBAL-ARPEGE-01-EUROPE-WCS`.
- Lecture GRIB : `cfgrib.open_datasets()` (niveaux mélangés) ; grille AROME France 0,01° :
  2801×1791, lon −12→16, lat 55,4→37,5.

### Autres
- **ECMWF Open Data** (CC-BY-4.0, commercial OK) : IFS/AIFS 0,25°, jusqu'à 15 j
  (IFS 00/12 : 3 h jusqu'à 144 h puis 6 h), rétention ~12 runs. `ecmwf-opendata`.
- **ERA5-Land** (0,1°, horaire, 1950→) : utile surtout pour le **rayonnement** et
  combler les trous. ERA5-HEAT (`derived-utci-historical`) : UTCI/MRT 0,25°, pour calibrer.
- **Landsat 8/9 C2 L2** : LST 30 m, `ST_B10 × 0.00341802 + 149` (K), passage ~10h30 UTC,
  pas de nuit, 8 j combinés. Gratuit via **Planetary Computer** (`landsat-c2-l2`,
  bande `lwir11`). **Earth Engine = payant en usage commercial.**
- **Sentinel-2** : pas de bande thermique (NDVI 10 m). **Sentinel-3 SLSTR LST** : 1 km,
  jour + nuit, via **openEO** Copernicus (reprojection côté serveur).
- **LCZ Cerema** : 93 territoires (dont DROM), shapefile Lambert-93 (EPSG:2154) + raster
  1,5 m ; champs `lcz, lcz_int, hre, are, bur, ror, bsr, war, ver, vhr` (hauteur, rapport
  H/L, part bâtie, imperméable, sol nu, eau, végétation, végétation haute – à confirmer).
- **Sat4BDNB** : indicateurs ICU par **IRIS**, été 2022 (1/06–31/08), **licence ODbL**
  (attention redistribution), + scénarios végétation/albédo (2026).
- **BDNB** (millésime 2026-02.a) : fiche par bâtiment (DPE, matériaux, hauteur, usage).
  Surtout résidentiel/tertiaire. **BD TOPO** : environnement (routes, végétation, tous
  bâtiments) – plus tard. DPE ADEME brut : contient l'**indicateur de confort d'été**.
- **Santé** : décès quotidiens INSEE (département) et fichier individuel (commune × jour,
  nominatif → agréger). Pour calibrer les seuils, pas comme cible.
- **AT/MP** : data.ameli.fr (open data depuis 09/2026), par code NAF, 2015-2024 → argument ROI.
- **SIRENE** : prospects (BTP NAF 41-43, logistique NAF 52).

## Prochaines étapes
1. Faire tourner l'onglet Stations sur de vraies données (dép. 13, période `latest`).
2. Onglet Urbain (LCZ + BDNB).
3. Onglet Satellite (Landsat médiane estivale).
4. AROME/ARPEGE (clé portail), ERA5 (clé CDS).
5. Croisement → Modèle (cible : T au site ; features : météo de référence + fiche statique
   du site ; validation par station exclue).
