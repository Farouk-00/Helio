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

## Structure

```
Accueil.py                  page d'accueil
pages/                      un fichier par onglet
core/config.py              chemins, identifiants des jeux de données
core/zones.py               départements métropole + outre-mer
core/ui.py                  sélecteur de zone partagé
core/sources/               un module par source (télécharger / convertir / lire)
data/raw, data/processed    stockage local (non versionné)
```

## Notes

- Heures : UTC pour la métropole (converties en heure de Paris à l'affichage),
  heure locale pour l'outre-mer.
- Qualité : l'option « valeurs validées uniquement » garde les codes Q = 0 ou 1.
