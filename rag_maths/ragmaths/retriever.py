"""Recherche hybride : BM25 (toujours) + embeddings (si `MODELE_EMBEDDING` est défini),
fusionnés par Reciprocal Rank Fusion.

BM25 est très bon sur les cours de maths (vocabulaire précis : « diagonalisable »,
« \\det », « polynôme caractéristique ») ; les embeddings aident sur les reformulations.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ingest import Chunk

MOTS_VIDES = set(
    "le la les un une des de du d l et ou en est sont que qui dans pour par sur au aux ce "
    "cette ces on a il elle se ne pas plus avec si alors soit tout tous toute on".split()
)


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in texte if not unicodedata.combining(c))


def tokeniser(texte: str) -> List[str]:
    # Les commandes LaTeX (\det, \ker, \lambda) sont gardées comme mots.
    jetons = re.findall(r"\\?[a-z0-9]+", normaliser(texte))
    return [j.lstrip("\\") for j in jetons if j.lstrip("\\") not in MOTS_VIDES and len(j) > 1]


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(d) for d in docs]
        self.longueurs = [len(d) for d in docs]
        self.moy = sum(self.longueurs) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, requete: List[str]) -> List[float]:
        res = []
        for d, L in zip(self.docs, self.longueurs):
            s = 0.0
            for t in requete:
                if t in d:
                    f = d[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.moy))
            res.append(s)
        return res


class Index:
    def __init__(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None):
        self.chunks = chunks
        self.par_id: Dict[str, Chunk] = {c.id: c for c in chunks}
        self.embeddings = embeddings
        self.bm25 = BM25([tokeniser(c.section + " " + c.texte) for c in chunks])

    # --- persistance -------------------------------------------------------------------
    def sauver(self, chemin: Path) -> None:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        data = {"chunks": [c.to_dict() for c in self.chunks], "embeddings": self.embeddings}
        chemin.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def charger(cls, chemin: Path) -> "Index":
        data = json.loads(chemin.read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in data["chunks"]], data.get("embeddings"))

    # --- recherche ---------------------------------------------------------------------
    def chercher(self, requete: str, k: int = 6, embed=None) -> List[Chunk]:
        s_bm25 = self.bm25.scores(tokeniser(requete))
        classements = [sorted(range(len(self.chunks)), key=lambda i: -s_bm25[i])]
        if self.embeddings and embed is not None:
            q = embed([requete])[0]
            s_emb = [_cos(q, e) for e in self.embeddings]
            classements.append(sorted(range(len(self.chunks)), key=lambda i: -s_emb[i]))
        # Reciprocal Rank Fusion (k=60, valeur standard).
        rrf: Dict[int, float] = {}
        for cl in classements:
            for rang, i in enumerate(cl):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rang)
        meilleurs = sorted(rrf, key=lambda i: -rrf[i])
        # Sans embeddings, on écarte les chunks sans aucun mot commun.
        if len(classements) == 1:
            meilleurs = [i for i in meilleurs if s_bm25[i] > 0]
        return [self.chunks[i] for i in meilleurs[:k]]

    def lire_section(self, nom: str, debut: int = 0, n: int = 25) -> Tuple[List[Chunk], int]:
        """Chunks (dans l'ordre du cours) dont « fichier > section » contient `nom`.

        Renvoie (chunks[debut:debut+n], nombre total) pour lire un long chapitre en plusieurs fois.
        """
        cle = normaliser(nom).strip()
        tous = [c for c in self.chunks if cle and cle in normaliser(f"{c.source} > {c.section}")]
        return tous[debut:debut + n], len(tous)

    def sections(self) -> List[str]:
        vues, res = set(), []
        for c in self.chunks:
            cle = f"{c.source} > {c.section}"
            if cle not in vues:
                vues.add(cle)
                res.append(cle)
        return res


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0
