"""Configuration lue dans l'environnement (fichier .env facultatif)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _charger_dotenv(chemin: Path) -> None:
    """Lecteur .env minimal (évite une dépendance)."""
    if not chemin.exists():
        return
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, val = ligne.split("=", 1)
        os.environ.setdefault(cle.strip(), val.strip().strip('"').strip("'"))


_charger_dotenv(RACINE / ".env")


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Identifiants OpenRouter : vérifier sur https://openrouter.ai/models?q=mistral
    modele: str = os.environ.get("MODELE", "mistralai/mistral-medium-3-5")
    modele_juge: str = os.environ.get("MODELE_JUGE", "mistralai/mistral-large-4-0")
    # Vide = recherche lexicale BM25 seule (aucun appel d'embedding).
    modele_embedding: str = os.environ.get("MODELE_EMBEDDING", "")
    dossier_cours: Path = Path(os.environ.get("DOSSIER_COURS", RACINE / "cours"))
    fichier_index: Path = Path(os.environ.get("FICHIER_INDEX", RACINE / "data" / "index.json"))
    taille_chunk: int = 1200  # caractères
    recouvrement: int = 200
    k: int = 6
    etapes_max: int = 6
    temperature: float = 0.2


CONFIG = Config()
