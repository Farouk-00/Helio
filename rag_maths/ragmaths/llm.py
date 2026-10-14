"""Client OpenRouter (API compatible OpenAI) pointé sur les modèles Mistral."""
from __future__ import annotations

from typing import List

from .config import CONFIG


def client():
    from openai import OpenAI

    if not CONFIG.api_key:
        raise RuntimeError("OPENROUTER_API_KEY manquante (voir .env.example).")
    return OpenAI(
        base_url=CONFIG.base_url,
        api_key=CONFIG.api_key,
        # En-têtes facultatifs : l'appli apparaît dans le tableau de bord OpenRouter.
        default_headers={"X-Title": "rag-maths"},
    )


def embedder(cli=None):
    """Renvoie une fonction textes -> vecteurs, ou None si aucun modèle d'embedding."""
    if not CONFIG.modele_embedding:
        return None
    cli = cli or client()

    def embed(textes: List[str]) -> List[List[float]]:
        res = []
        for i in range(0, len(textes), 64):
            r = cli.embeddings.create(model=CONFIG.modele_embedding, input=textes[i:i + 64])
            res.extend(d.embedding for d in r.data)
        return res

    return embed
