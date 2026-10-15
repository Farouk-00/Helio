"""Client Mistral : API Mistral directe ou OpenRouter (toutes deux au format OpenAI)."""
from __future__ import annotations

from typing import List

from .config import CONFIG


def client():
    from openai import OpenAI

    if not CONFIG.api_key:
        var = "MISTRAL_API_KEY" if CONFIG.fournisseur == "mistral" else "OPENROUTER_API_KEY"
        raise RuntimeError(f"{var} manquante (voir .env.example).")
    return OpenAI(
        base_url=CONFIG.base_url,
        api_key=CONFIG.api_key,
        max_retries=6,  # 429 (limite de débit) et erreurs réseau : nouvelles tentatives espacées
        # Nom de l'appli dans le tableau de bord OpenRouter (ignoré par Mistral).
        default_headers={"X-Title": "rag-maths"},
    )


def texte(contenu) -> str:
    """Contenu d'un message -> texte.

    OpenRouter renvoie une chaîne ; l'API Mistral peut renvoyer une liste de blocs
    (`{"type": "thinking", ...}`, `{"type": "text", "text": ...}`) : on ne garde que le texte.
    """
    if contenu is None:
        return ""
    if isinstance(contenu, str):
        return contenu
    morceaux = []
    for bloc in contenu:
        if isinstance(bloc, str):
            morceaux.append(bloc)
            continue
        get = bloc.get if isinstance(bloc, dict) else lambda k, d=None: getattr(bloc, k, d)
        if get("type", "text") == "text":
            morceaux.append(get("text") or "")
    return "".join(morceaux)


def embedder(cli=None):
    """Renvoie une fonction textes -> vecteurs, ou None si aucun modèle d'embedding."""
    if not CONFIG.modele_embedding or not CONFIG.api_key:
        return None  # pas de clé (ou désactivé) : BM25 seul
    cli = cli or client()

    def embed(textes: List[str]) -> List[List[float]]:
        res = []
        for i in range(0, len(textes), 32):
            r = cli.embeddings.create(model=CONFIG.modele_embedding, input=textes[i:i + 32])
            res.extend(d.embedding for d in r.data)
        return res

    return embed
