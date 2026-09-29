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
| Stations | ✅ testé sur données réelles (dép. 13, `latest`) | Météo-France climatologie horaire (data.gouv) |
| Urbain | ✅ testé par Foucault, fusionné dans `main` | LCZ Cerema, BDNB (sans clé) |
| Satellite | ✅ Landsat testé par Foucault, fusionné dans `main` ; Sentinel-3 nuit à faire | Landsat via Planetary Computer, Sentinel-2/3 via openEO |
| AROME / ARPEGE | ✅ fusionné, testé en réel par Foucault (T, HR, rayonnement, pluie) | API ciblée Modèles (clés `METEOFRANCE_AROME_KEY`, `METEOFRANCE_ARPEGE_KEY`, en-tête `apikey`) |
| ERA5 | ✅ fusionné, clé CDS de Foucault en place | Copernicus CDS (`~/.cdsapirc`), jeux « time-series » |
| Croisement | ✅ fusionné : fiche fixe + table horaire, export pour le modèle | toutes les sources + altitude IGN, distance à la mer IGN, NDVI Sentinel-2 |
| Modèle | ✅ codé (branche `modele`), chaîne testée (ERA5 simulé) ; à lancer avec le vrai ERA5 | LightGBM (repli scikit-learn), cible = T station − ERA5, stations exclues + test temporel |

### Vérifié sur données réelles (09/2026)
- Stations : `GLO` à l'horodatage HH UTC = cumul de HH-1 à HH (profil diurne de Marignane,
  midi solaire 11 h 40 UTC). Marignane (13054001) a T, TD, U, vent, PMER, N (tri-horaire), GLO :
  bonne référence « rurale » ; Marseille-Obs (13055001) : T seulement.
- Stations : noms `H_<dep>_<periode>.csv.gz` et décimale `.` confirmés (dép. 13, `latest-2025-2026` :
  20 stations, 258 k lignes, 11 Mo -> 2,5 Mo Parquet). `GLO` : 1 station sur 20, tout en code 9.
  Le fichier `latest` s'arrêtait au 24/06/2026 (fin septembre 2026).

### À vérifier au premier vrai run
- LCZ : structure réelle du zip Cerema (nom du shapefile, nom de la colonne de classe, sens des
  champs `hre, are, bur, ror, bsr, war, ver, vhr`). Le code choisit le plus gros `.shp` nommé
  `*lcz*` et la première colonne dont les valeurs se décodent en classe LCZ.
- Rendu des cartes plotly (MapLibre) : non vérifiable depuis le cloud (Urbain validé en local).
- Landsat : dans le composite de Marseille, le Vieux-Port ressort masqué (NaN) ; à comprendre
  (bits `qa_pixel` sur l'eau ?) si on veut l'eau.
- ERA5 (premier téléchargement réel) : forme exacte du CSV renvoyé par les jeux « time-series »
  (un CSV ou un zip, noms courts `t2m` ou longs), et si ERA5-Land y est cumulé depuis 00 UTC.
  Le lecteur gère tous ces cas (tests synthétiques) et détecte le cumul sur `strd`.

- Altitude : `data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json`, ressource
  `ign_rge_alti_wld` (72,5 m à Marseille-Obs pour 75 m annoncés).
- Distance à la mer : WFS Géoplateforme `BDCARTO_V5:limite_terre_mer` (BD TOPO aussi dispo :
  `BDTOPO_V3:limite_terre_mer`). **BBOX sans ambiguïté seulement avec le système explicite**
  `…,urn:ogc:def:crs:OGC:1.3:CRS84` (lon, lat) ; sans lui, 0 objet. Inclut les lagunes
  ouvertes (étang de Berre : Marignane à 0,8 km). Au-delà de ~50 km : Natural Earth 1:10 M
  (écart ~1 km : Marseille-Obs 3,5 km contre 2,0 km avec l'IGN).
- Croisement Marseille-Obs (JJA 2025 + 2026) : RMSE station voisine 1,91 °C, ERA5 1,74 °C
  (ERA5 simulé dans le test local, à refaire avec le vrai ERA5).

- AROME / ARPEGE (diagnostic réel du 29/09/2026) : 5 610 couvertures AROME (46 paramètres), 4 621 ARPEGE
  (94) ; heure du run écrite `T03.00.00Z` ; runs AROME toutes les 3 h, 52 échéances (+51 h) ; le run
  le plus récent est publié progressivement (ARPEGE 06 h : 1 seule échéance au moment du test).
  Cumuls `PT1H` présents pour rayonnement et pluie (aussi PT3H… P1D, P2D). Rayonnement en J/m² sur
  l'heure : ~640 W/m² à 12 h UTC fin septembre à Marseille, cohérent.
- Environnement de Foucault : `pip` pointait sur le Python système 3.9 (paquets en site utilisateur) ;
  LightGBM sur Mac exige `brew install libomp` (la roue n'embarque pas OpenMP).

### Premier modèle réel (13, ERA5-Land 2025, 17 stations sur 19, LightGBM, 29/09/2026)
RMSE °C, erreur = prévu − mesuré ; sans Landsat, LCZ manquante pour Arles, Istres, Eyragues.
| périmètre | ERA5 brut | modèle (stations exclues) | modèle (test temporel oct.-déc.) |
|---|---|---|---|
| toutes les heures | 2,01 | 1,71 | 1,45 (ERA5 1,88) |
| heures ≥ 30 °C | 3,05 | 2,39 | – |
| Tmin journalière | 2,11 | 1,69 | 1,53 (ERA5 2,14) |
| Tmax journalière | 2,45 | 1,90 | 1,13 (ERA5 2,08) |
- ERA5 trop chaud la nuit (+0,5 °C), trop froid l'après-midi (−1,2 °C à 14 h) ; biais du modèle ~0 à toute heure.
- Importance (gain) : era5_GHI >> jour_annee > era5_WS > lcz_ror > lcz_ver > era5_RH > era5_WD…
- **Méthode, après ce run** : `jour_annee` retiré des variables (avec une seule année, il apprend
  l'erreur d'ERA5 d'une date donnée, vue sur les autres stations : score « stations exclues »
  optimiste). Ajout du test **stations exclues ET période future** (le plus honnête). Pour juger la
  canicule : couper avant un été (ERA5 2026 + mesures jusqu'à fin juin 2026 → été 2026 à compléter).

### Prototype UWG (09/2026) – voir `prototypes/uwg_marseille/README.md`
- Marignane -> Marseille-Observatoire, JJA 2025, sans calage : RMSE horaire = référence naïve
  (1,49 °C) mais **Tmin 1,45 -> 0,82 °C** (avec correction d'altitude). UWG surestime l'îlot
  nocturne (~+1 °C) et rate la brise de mer de l'après-midi (ville 1-1,5 °C plus fraîche à 15 h).
  => variable physique pour le modèle, pas un prédicteur autonome.
- UWG diverge parfois à `dtsim=300` s (bureaux « pre80 ») : relancer à 150 s. Année non
  bissextile, EPW à 3 profondeurs de sol obligatoires. GPL-3.
- Ladybug Tools (plugins Rhino payant) et SOLENE-microclimat (CFD, Linux, par quartier) :
  écartés pour la prévision ; `ladybug-comfort` utile plus tard pour l'UTCI.

## Décisions prises
- Langage : Python + Streamlit. Stockage local Parquet/NetCDF, déploiement plus tard sur
  Google Cloud (Cloud Run + Cloud Storage + Cloud Scheduler, région europe-west9).
- **Archivage des prévisions** : amorcé au site (chaque run téléchargé dans l'onglet AROME / ARPEGE
  est gardé dans `data/processed/prevision/`) ; l'archivage automatique reste à faire (mais c'est la seule donnée
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
  combler les trous. Jeux CDS « time-series » (au point, rapides) vérifiés :
  `reanalysis-era5-land-timeseries` (0,1°, ~6 j de retard) et
  `reanalysis-era5-single-levels-timeseries` (0,25°, + `total_cloud_cover`, rayonnement direct
  `fdir`). Requête : `{"variable": [...], "location": {"longitude": x, "latitude": y},
  "date": ["AAAA-MM-JJ/AAAA-MM-JJ"], "data_format": "csv"}` ; sans clé : 401. CC-BY 4.0. ERA5-HEAT (`derived-utci-historical`) : UTCI/MRT 0,25°, pour calibrer.
- **Landsat 8/9 C2 L2** : LST 30 m, `ST_B10 × 0.00341802 + 149` (K), passage ~10h30 UTC,
  pas de nuit, 8 j combinés. Gratuit via **Planetary Computer** (`landsat-c2-l2`,
  bande `lwir11`). **Earth Engine = payant en usage commercial.**
  Vérifié : STAC sans clé (`pystac-client` + `planetary_computer.sign_inplace`), le filtre
  `query` n'est pas supporté (filtrer nuages/plateforme côté client). Masque `qa_pixel` bits 0-5.
  Marseille été 2025 : 18 scènes (≤ 30 % nuages), 17 dates exploitables, lecture ~26 s pour 1 km.
- **Sentinel-2** : pas de bande thermique (NDVI 10 m). **Sentinel-3 SLSTR LST** : 1 km,
  jour + nuit, via **openEO** Copernicus (reprojection côté serveur).
- **LCZ Cerema** : 93 territoires (dont DROM), zip par territoire (20-230 Mo) sur data.gouv
  (jeu `6641c562e5acdb35c0e6051d`), shapefile Lambert-93 (EPSG:2154) + raster 1,5 m ; champs `lcz, lcz_int, hre, are, bur, ror, bsr, war, ver, vhr` (hauteur, rapport
  H/L, part bâtie, imperméable, sol nu, eau, végétation, végétation haute – à confirmer).
  Service ArcGIS public du Cerema (`cartagene.cerema.fr/server/rest/services/Hosted/…`) :
  `aires_urbaines_3857` (93 contours + lien data.gouv, requêtable), `statistiques_commune_3857`
  (part de chaque LCZ par commune, 34 955 communes), tuiles nationales `l_lcz_spot_000_2022_tl`
  (PNG, **zoom ≤ 14** seulement). Pas de LCZ 10 en France.
- **Sat4BDNB** : indicateurs ICU par **IRIS**, été 2022 (1/06–31/08), **licence ODbL**
  (attention redistribution), + scénarios végétation/albédo (2026).
- **BDNB** (millésime 2026-02.a) : fiche par bâtiment (DPE, matériaux, hauteur, usage).
  API ouverte `api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet` (PostgREST, ~140 colonnes) :
  **sans clé, 10 lignes par requête et 120 requêtes/min** ; `/bbox?xmin&ymin&xmax&ymax` en
  **Lambert-93** ; `Prefer: count=exact` donne le total. **Aucun bâtiment outre-mer** (971, 974).
  Surtout résidentiel/tertiaire. **BD TOPO** : environnement (routes, végétation, tous
  bâtiments) – plus tard. DPE ADEME brut : contient l'**indicateur de confort d'été**.
- **Santé** : décès quotidiens INSEE (département) et fichier individuel (commune × jour,
  nominatif → agréger). Pour calibrer les seuils, pas comme cible.
- **AT/MP** : data.ameli.fr (open data depuis 09/2026), par code NAF, 2015-2024 → argument ROI.
- **SIRENE** : prospects (BTP NAF 41-43, logistique NAF 52).

## Prochaines étapes
1. ~~Stations~~, ~~Urbain~~, ~~Satellite~~, ~~ERA5 + UWG~~, ~~Croisement~~, ~~AROME / ARPEGE~~ (fusionnés).
   Modèle : branche `modele-propre` (ne pas fusionner `modele`, qui contient un fichier de test) ;
   relancer avec Landsat + LCZ complètes, lire le test stations exclues + période future.
2. Géocodage : Géoplateforme (`data.geopf.fr/geocodage/search`), api-adresse est en fin de vie.
3. AROME-OM (outre-mer) ; archivage automatique des runs (tâche planifiée) ; UWG forcé par
   ERA5-Land sur d'autres villes.
4. Prévision J+1/J+2 : appliquer le modèle avec AROME comme météo de fond (écart de distribution
   ERA5 -> AROME à mesurer : réentraîner sur l'archive AROME dès qu'elle est assez longue).
5. Plus de stations (départements voisins, réseaux urbains type Météo Marseille / Montpellier) ;
   variables UWG (écart UWG − rural) et Landsat / LCZ dans le jeu.
