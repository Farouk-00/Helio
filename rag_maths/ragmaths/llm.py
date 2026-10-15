"""Client Mistral : API Mistral directe ou OpenRouter (toutes deux au format OpenAI)."""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, List

from .config import CONFIG


def client():
    from openai import OpenAI

    if not CONFIG.api_key:
        var = "MISTRAL_API_KEY" if CONFIG.fournisseur == "mistral" else "OPENROUTER_API_KEY"
        raise RuntimeError(f"{var} manquante (voir .env.example).")
    return OpenAI(
        base_url=CONFIG.base_url,
        api_key=CONFIG.api_key,
        max_retries=0,  # les reprises sont gérées par `appeler` (attentes plus longues)
        # Nom de l'appli dans le tableau de bord OpenRouter (ignoré par Mistral).
        default_headers={"X-Title": "rag-maths"},
    )


class _Limiteur:
    """Espace les requêtes : au plus `REQ_PAR_SEC` par seconde, tous threads confondus.

    Mistral limite le débit par espace de travail (requêtes/s), indépendamment des crédits.
    """

    def __init__(self) -> None:
        self.verrou, self.prochain = threading.Lock(), 0.0

    def attendre(self) -> None:
        if CONFIG.req_par_sec <= 0:
            return
        with self.verrou:
            maintenant = time.monotonic()
            depart = max(maintenant, self.prochain)
            self.prochain = depart + 1.0 / CONFIG.req_par_sec
        time.sleep(max(0.0, depart - maintenant))


LIMITEUR = _Limiteur()


def appeler(fn: Callable, *args, tentatives: int = 8, **kwargs):
    """Appel API limité en débit, avec reprises sur 429 / 5xx / réseau (attente exponentielle)."""
    for essai in range(tentatives):
        LIMITEUR.attendre()
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # openai.RateLimitError, APIConnectionError, InternalServerError…
            code = getattr(e, "status_code", None)
            reessayable = code in (429, 500, 502, 503, 504) or type(e).__name__ in (
                "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")
            if not reessayable or essai == tentatives - 1:
                raise
            time.sleep(min(60.0, 2.0 ** essai) * (1 + random.random() / 2))


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
            r = appeler(cli.embeddings.create, model=CONFIG.modele_embedding, input=textes[i:i + 32])
            res.extend(d.embedding for d in r.data)
        return res

    return embed
