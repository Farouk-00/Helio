# Prototype UWG – Marseille, été 2025

**Question.** UWG (Urban Weather Generator, MIT, `pip install uwg`, GPL-3), forcé par la
station « rurale » de **Marignane** (aéroport, 6 m), retrouve-t-il la température mesurée
en ville à **Marseille-Observatoire** (Palais Longchamp, 75 m) ?

Lancement (depuis la racine du dépôt, après avoir téléchargé la période `latest` du 13
dans l'onglet Stations) :

```bash
python -m prototypes.uwg_marseille.run
```

Sorties dans `data/prototypes/uwg_marseille/` : EPW de forçage et EPW urbain, CSV horaire,
`metriques.json`, `resultats.html` (profil de l'îlot de chaleur et séries).

## Méthode

| Étape | Choix |
|---|---|
| Forçage | EPW 2025 construit depuis les observations horaires de Marignane (`core/epw.py`) : T, TD, U, vent, PMER, nébulosité (tri-horaire, interpolée), rayonnement global `GLO` (cumul de l'heure qui se termine à l'horodatage, vérifié sur le profil diurne). Direct / diffus par Erbs, infrarouge par la formule EnergyPlus. Heure locale normale UTC+1. |
| Morphologie (500 m) | BDNB, 1 273 bâtiments : hauteur moyenne pondérée par l'emprise **13,4 m**, emprise bâtie **0,37**, façades / sol **1,38**. Usages (emprise) : 65 % résidentiel, 31 % tertiaire. |
| Végétation (500 m) | Sentinel-2 L2A du 29/06/2025 (0 % de nuages), NDVI corrigé du décalage −1000 : arbres (NDVI ≥ 0,5) **13 %**, herbe (0,3–0,5) **19 %**. |
| UWG | Zone ASHRAE 3A, bâtiments de référence « pre80 » (65 % logements, 31 % bureaux, 4 % commerces), autres paramètres par défaut, **aucun calage**. Du 25/05 au 31/08, 7 jours de mise en route écartés. Pas de 300 s : divergence numérique au 8 juillet → relancé à **150 s**. |
| Évaluation | 2 100 heures (juin – août), mesures validées (codes 0 et 1). Référence naïve : « la ville a la température de Marignane ». Variante : UWG corrigé de l'altitude (−0,0065 °C/m × 69 m = −0,45 °C). |

## Résultats (erreur = prévu − mesuré à Marseille-Observatoire, °C)

| | Horaire RMSE | Horaire biais | Nuit RMSE | Tmin jour. RMSE | Tmax jour. RMSE |
|---|---|---|---|---|---|
| Référence naïve (T Marignane) | 1,49 | +0,24 | 1,08 | 1,45 | 1,74 |
| UWG brut | 1,70 | +0,99 | 1,54 | 1,09 | 1,83 |
| **UWG + altitude** | **1,49** | +0,54 | 1,25 | **0,82** | **1,58** |

Sur les 10 jours les plus chauds : erreur absolue sur Tmin 1,63 → **1,13** °C, sur Tmax
2,23 → **1,95** °C (naïf → UWG + altitude).

Profil moyen de l'îlot de chaleur (ville − Marignane), heure légale :

| heure | 0 h | 3 h | 6 h | 9 h | 12 h | 15 h | 18 h | 21 h |
|---|---|---|---|---|---|---|---|---|
| mesuré | +0,2 | +0,6 | +0,7 | −0,3 | −0,8 | −1,4 | −1,1 | 0,0 |
| UWG | +1,4 | +1,7 | +1,5 | −0,5 | −0,2 | +0,2 | +0,4 | +1,6 |

Calage horaire vérifié : l'erreur est minimale sans décalage (RMSE 1,70 à 0 h contre
1,77 à −1 h et 2,08 à +1 h).

## Lecture

- **UWG apporte de l'information sur les minimales** (nuits, l'enjeu sanitaire des
  canicules) : RMSE sur Tmin 1,45 → 0,82 °C une fois l'altitude corrigée.
- **Il surestime l'îlot nocturne** d'environ 1 °C avec les paramètres par défaut (chaleur
  anthropique, bâtiments de référence américains, inertie).
- **Il rate l'après-midi** : en été, Marseille-Observatoire est 1 à 1,5 °C *plus frais*
  que Marignane vers 15 h (brise de mer, altitude, jardin de l'Observatoire). UWG n'a pas
  de brise de mer : c'est un effet que le modèle statistique devra apprendre (distance à
  la mer, direction du vent).
- Conclusion : **pas un prédicteur autonome**, mais une **variable physique utile** pour
  LightGBM (écart UWG − rural), surtout la nuit. À ne pas caler sur ce seul site / été.

## Limites

- Un seul site et un seul été ; la station de l'Observatoire est dans un jardin, plus
  verte que la moyenne des 500 m.
- Paramètres UWG non calés (volontairement) ; seuils NDVI arbres / herbe arbitraires.
- UWG a divergé à 300 s : prévoir la relance à pas plus fin en production.
- Licence GPL-3 : usage interne / service en ligne sans problème ; redistribuer un
  logiciel qui l'embarque imposerait la GPL.

## Suite proposée

1. Rejouer sur d'autres couples ville / référence (Lyon, Toulouse, Paris) et d'autres étés.
2. Forcer par ERA5-Land (puis AROME en prévision) au lieu d'une station, pour tout site.
3. Brancher l'écart UWG − rural comme variable dans l'onglet Modèle.
