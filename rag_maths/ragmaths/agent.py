"""Agent de résumé : boucle d'appels d'outils (function calling) sur le cours indexé.

Deux modes, pour pouvoir les comparer à l'évaluation :
- `agent` : le modèle décide lui-même quoi chercher (plusieurs requêtes, plan du cours) ;
- `rag`   : une seule recherche avec la demande de l'utilisateur, puis un seul appel.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import CONFIG
from .latex import ENVS_AUTORISES, extraire_corps
from .retriever import Index

SYSTEME = f"""Tu es un assistant qui rédige des fiches de révision de mathématiques \
(niveau licence / classes préparatoires) à partir du cours de l'étudiant.

Règles :
1. Tu t'appuies UNIQUEMENT sur les extraits du cours obtenus avec l'outil `chercher_cours`. \
N'ajoute aucun résultat absent du cours ; si une notion demandée n'y est pas, écris \
`% ABSENT DU COURS : <notion>`.
2. Commence par `plan_du_cours` si la demande est large, puis fais plusieurs recherches ciblées \
(définitions, théorèmes, méthodes, exemples) avant de rédiger.
3. Garde les énoncés exacts (hypothèses comprises) ; résume les démonstrations en une ou deux \
lignes d'idée clé, ne les recopie pas.
4. Après chaque définition, théorème ou méthode, cite la source avec `\\src{{cID}}` \
(identifiant du chunk, ex. `\\src{{c12}}`).
5. Format de sortie : UNIQUEMENT le corps d'un document LaTeX (pas de préambule, pas de \
\\begin{{document}}), avec \\section / \\subsection et les environnements : {ENVS_AUTORISES}. \
Les macros \\R, \\C, \\K, \\N sont définies. Pas de package supplémentaire.
6. Termine par une \\section{{À retenir}} de 5 points maximum (itemize)."""

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "chercher_cours",
            "description": "Recherche des extraits du cours de maths (recherche plein texte + sémantique).",
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {"type": "string", "description": "Mots-clés, ex. 'théorème spectral matrice symétrique'"},
                    "k": {"type": "integer", "description": "Nombre d'extraits (défaut 6, max 10)"},
                },
                "required": ["requete"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_du_cours",
            "description": "Liste les fichiers et sections du cours indexé.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class Resultat:
    corps: str
    chunks_vus: List[str] = field(default_factory=list)
    requetes: List[str] = field(default_factory=list)
    etapes: int = 0
    tokens_entree: int = 0
    tokens_sortie: int = 0
    duree_s: float = 0.0


def _formater(chunks) -> str:
    return "\n\n".join(f"[{c.id}] ({c.source} > {c.section})\n{c.texte}" for c in chunks)


class Agent:
    def __init__(self, index: Index, cli, modele: str = "", embed: Optional[Callable] = None):
        self.index, self.cli, self.embed = index, cli, embed
        self.modele = modele or CONFIG.modele

    def _appel(self, res: Resultat, **kw):
        r = self.cli.chat.completions.create(model=self.modele, temperature=CONFIG.temperature, **kw)
        if getattr(r, "usage", None):
            res.tokens_entree += r.usage.prompt_tokens or 0
            res.tokens_sortie += r.usage.completion_tokens or 0
        return r.choices[0].message

    def _outil(self, nom: str, args: Dict, res: Resultat) -> str:
        if nom == "plan_du_cours":
            return "\n".join(self.index.sections())
        if nom == "chercher_cours":
            req = str(args.get("requete", ""))
            k = max(1, min(int(args.get("k") or CONFIG.k), 10))
            res.requetes.append(req)
            chunks = self.index.chercher(req, k=k, embed=self.embed)
            res.chunks_vus += [c.id for c in chunks if c.id not in res.chunks_vus]
            return _formater(chunks) or "Aucun résultat."
        return f"Outil inconnu : {nom}"

    def resumer(self, demande: str, mode: str = "agent") -> Resultat:
        t0, res = time.time(), Resultat(corps="")
        if mode == "rag":
            contexte = self._outil("chercher_cours", {"requete": demande, "k": 10}, res)
            msgs = [
                {"role": "system", "content": SYSTEME.replace("obtenus avec l'outil `chercher_cours`", "fournis ci-dessous")},
                {"role": "user", "content": f"Extraits du cours :\n\n{contexte}\n\nDemande : {demande}"},
            ]
            res.corps = extraire_corps(self._appel(res, messages=msgs).content or "")
            res.etapes, res.duree_s = 1, time.time() - t0
            return res

        msgs: List[Dict] = [{"role": "system", "content": SYSTEME}, {"role": "user", "content": demande}]
        for etape in range(CONFIG.etapes_max):
            derniere = etape == CONFIG.etapes_max - 1
            msg = self._appel(res, messages=msgs, tools=OUTILS, tool_choice="none" if derniere else "auto")
            res.etapes = etape + 1
            if not msg.tool_calls:
                res.corps = extraire_corps(msg.content or "")
                break
            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                        for t in msg.tool_calls
                    ],
                }
            )
            for t in msg.tool_calls:
                try:
                    args = json.loads(t.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": self._outil(t.function.name, args, res)})
        res.duree_s = time.time() - t0
        return res
