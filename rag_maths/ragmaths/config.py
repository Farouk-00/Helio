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


# Deux fournisseurs, même format d'API (compatible OpenAI) :
# - "mistral"    : API Mistral directe (La Plateforme, console.mistral.ai), noms courts ;
# - "openrouter" : passerelle OpenRouter, noms préfixés « mistralai/ ».
FOURNISSEUR = os.environ.get("FOURNISSEUR", "openrouter").lower()
_DEFAUTS = {
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                "mistral-medium-latest", "mistral-large-latest", "mistral-embed"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "mistralai/mistral-medium-3-5", "mistralai/mistral-large-4-0", "mistralai/mistral-embed-2312"),
}
if FOURNISSEUR not in _DEFAUTS:
    raise ValueError("FOURNISSEUR = mistral | openrouter")
_URL, _VAR_CLE, _MODELE, _JUGE, _EMBED = _DEFAUTS[FOURNISSEUR]


@dataclass
class Config:
    fournisseur: str = FOURNISSEUR
    api_key: str = field(default_factory=lambda: os.environ.get(_VAR_CLE, ""))
    base_url: str = os.environ.get("BASE_URL", _URL)
    modele: str = os.environ.get("MODELE", _MODELE)
    modele_juge: str = os.environ.get("MODELE_JUGE", _JUGE)
    # MODELE_EMBEDDING= (vide) : recherche lexicale BM25 seule, aucun appel d'embedding.
    modele_embedding: str = os.environ.get("MODELE_EMBEDDING", _EMBED)
    dossier_cours: Path = Path(os.environ.get("DOSSIER_COURS", RACINE / "cours"))
    fichier_index: Path = Path(os.environ.get("FICHIER_INDEX", RACINE / "data" / "index.json"))
    taille_chunk: int = 1200  # caractères
    recouvrement: int = 200
    k: int = 6
    etapes_max: int = int(os.environ.get("ETAPES_MAX", 8))
    temperature: float = 0.2


CONFIG = Config()
