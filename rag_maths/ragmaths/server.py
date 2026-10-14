"""API HTTP minimale (FastAPI) pour faire tourner l'agent sur le VPS.

POST /resume  {"demande": "...", "mode": "agent"}  -> {"latex": "...", "chunks": [...], ...}
GET  /health
Authentification : en-tête `Authorization: Bearer $API_TOKEN` (si API_TOKEN est défini).
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .agent import Agent
from .config import CONFIG
from .latex import document
from .llm import client, embedder
from .retriever import Index

app = FastAPI(title="rag-maths")
_etat = {}


def _agent() -> Agent:
    if "agent" not in _etat:
        cli = client()
        _etat["agent"] = Agent(Index.charger(CONFIG.fichier_index), cli, embed=embedder(cli))
    return _etat["agent"]


def _auth(authorization: Optional[str] = Header(None)) -> None:
    jeton = os.environ.get("API_TOKEN")
    if jeton and authorization != f"Bearer {jeton}":
        raise HTTPException(401, "jeton invalide")


class Demande(BaseModel):
    demande: str
    mode: str = "agent"
    titre: str = ""


@app.get("/health")
def health():
    return {"ok": True, "index": CONFIG.fichier_index.exists(), "modele": CONFIG.modele}


@app.post("/resume", dependencies=[Depends(_auth)])
def resume(d: Demande):
    if d.mode not in ("agent", "rag"):
        raise HTTPException(400, "mode = agent | rag")
    res = _agent().resumer(d.demande, mode=d.mode)
    return {
        "latex": document(res.corps, d.titre or d.demande),
        "chunks": res.chunks_vus,
        "requetes": res.requetes,
        "tokens": res.tokens_entree + res.tokens_sortie,
        "duree_s": round(res.duree_s, 1),
    }
