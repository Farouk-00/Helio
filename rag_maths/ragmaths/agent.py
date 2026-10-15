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
from .llm import texte
from .retriever import Index

SYSTEME = f"""Tu es un assistant qui rédige des fiches de révision de mathématiques \
(niveau licence / classes préparatoires) à partir du cours de l'étudiant.

Règles :
1. Tu t'appuies UNIQUEMENT sur les extraits du cours obtenus avec les outils. \
N'ajoute aucun résultat absent du cours ; si une notion demandée n'y est pas, écris \
`% ABSENT DU COURS : <notion>`.
2. Pour un chapitre ou une section entière : `plan_du_cours`, puis `lire_section` sur la ou \
les sections concernées (texte intégral, dans l'ordre). Pour une notion précise : \
`chercher_cours`. Tu peux appeler plusieurs outils dans un même tour. Ne fais pas plus de \
recherches que nécessaire, puis rédige.
3. Garde les énoncés exacts (hypothèses comprises) ; résume les démonstrations en une ou deux \
lignes d'idée clé, ne les recopie pas.
4. Après chaque définition, théorème ou méthode, cite la source avec `\\src{{cID}}` \
(identifiant du chunk, ex. `\\src{{c12}}`).
5. Format de sortie : UNIQUEMENT le corps d'un document LaTeX (pas de préambule, pas de \
\\begin{{document}}), avec \\section / \\subsection et les environnements : {ENVS_AUTORISES}. \
Les macros \\R, \\C, \\K, \\N sont définies. Pas de package supplémentaire. N'écris jamais \
d'appel d'outil ou de JSON dans ta réponse finale.
6. Termine par une \\section{{À retenir}} de 5 points maximum (itemize).
7. Les consignes de l'utilisateur sur le contenu (pas de preuves, pas d'exemples, longueur…) \
priment sur ces règles par défaut."""

CONSIGNE_FIN = (
    "Les recherches sont terminées. Rédige MAINTENANT la fiche complète en LaTeX à partir des "
    "extraits obtenus, sans appeler d'outil ni écrire de JSON."
)

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
            "name": "lire_section",
            "description": (
                "Renvoie le texte intégral, dans l'ordre, des sections dont le nom contient `section` "
                "(ex. '1.2 Concentration' ou '1 Mathematical Preliminaries', noms donnés par plan_du_cours). "
                "Idéal pour résumer un chapitre entier. 25 extraits max par appel : utiliser `debut` pour la suite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "Nom (ou début de nom) de section"},
                    "debut": {"type": "integer", "description": "Indice du 1er extrait (défaut 0)"},
                },
                "required": ["section"],
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


def _pas_une_fiche(contenu: str) -> bool:
    """Réponse vide ou JSON (appel d'outil écrit en texte) au lieu d'une fiche LaTeX."""
    c = contenu.strip().strip("`").strip()
    return not c or (c[:1] in "{[" and c[-1:] in "}]" and "\\section" not in c)


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
        if nom == "lire_section":
            nom_section, debut = str(args.get("section", "")), max(0, int(args.get("debut") or 0))
            res.requetes.append(f"[section] {nom_section}")
            chunks, total = self.index.lire_section(nom_section, debut)
            res.chunks_vus += [c.id for c in chunks if c.id not in res.chunks_vus]
            if not chunks:
                return "Aucune section ne correspond : vérifier le nom avec plan_du_cours."
            fin = debut + len(chunks)
            suite = f"\n\n[{fin}/{total} extraits lus : lire_section(debut={fin}) pour la suite]" if fin < total else ""
            return _formater(chunks) + suite
        return f"Outil inconnu : {nom}"

    def _executer_outils(self, msg, msgs: List[Dict], res: Resultat) -> None:
        msgs.append(
            {
                "role": "assistant",
                "content": texte(msg.content),
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

    def resumer(self, demande: str, mode: str = "agent") -> Resultat:
        t0, res = time.time(), Resultat(corps="")
        if mode == "rag":
            contexte = self._outil("chercher_cours", {"requete": demande, "k": 10}, res)
            msgs = [
                {"role": "system", "content": SYSTEME.replace("obtenus avec les outils", "fournis ci-dessous")},
                {"role": "user", "content": f"Extraits du cours :\n\n{contexte}\n\nDemande : {demande}"},
            ]
            res.corps = extraire_corps(texte(self._appel(res, messages=msgs).content))
            res.etapes, res.duree_s = 1, time.time() - t0
            return res

        msgs: List[Dict] = [{"role": "system", "content": SYSTEME}, {"role": "user", "content": demande}]
        relance = False
        while True:
            derniere = relance or res.etapes >= CONFIG.etapes_max - 1
            if derniere and msgs[-1]["role"] == "tool":
                # L'API Mistral refuse un message « user » juste après « tool » : la consigne
                # de fin est donc ajoutée au dernier résultat d'outil.
                msgs[-1]["content"] += "\n\n" + CONSIGNE_FIN
            msg = self._appel(res, messages=msgs, tools=OUTILS, tool_choice="none" if derniere else "auto")
            res.etapes += 1
            if msg.tool_calls and not derniere:
                self._executer_outils(msg, msgs, res)
                continue
            contenu = texte(msg.content)
            if _pas_une_fiche(contenu) and not relance:
                # Le modèle a écrit un appel d'outil en texte (ou rien) : une relance explicite.
                msgs += [{"role": "assistant", "content": contenu or "…"},
                         {"role": "user", "content": CONSIGNE_FIN}]
                relance = True
                continue
            res.corps = extraire_corps(contenu)
            break
        res.duree_s = time.time() - t0
        return res
