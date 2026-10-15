"""Évaluation : un score composite sur un petit jeu de cas annotés à la main.

Pour chaque cas (`eval/cas.jsonl`) :
    {"id": ..., "demande": ..., "notions": [["diagonalisable"], ["polynôme caractéristique", "chi_A"]],
     "formules": ["\\chi_A(X)=\\det(XI_n-A)"], "sections_attendues": ["Diagonalisation"]}

Métriques (toutes dans [0, 1]) :
- latex       : compile (tectonic/pdflatex) ; sinon contrôle statique (accolades, envs, $).
- couverture  : part des notions attendues présentes (chaque notion = liste de synonymes).
- formules    : part des formules clés retrouvées (après normalisation du LaTeX).
- citations   : 0,5 × (citations qui pointent vers un chunk réellement lu)
                + 0,5 × (part des énoncés definition/theoreme/... suivis d'un \\src{}).
- retrieval   : part des sections attendues dont au moins un chunk a été lu (rappel).
- fidelite    : LLM-juge (autre modèle, voir juge.py) : chaque énoncé est vérifié seul contre
                les chunks qu'il cite ; score = part des énoncés corrects. L'indicateur
                principal est le NOMBRE D'ERREURS (colonne « erreurs », objectif 0).
                Désactivable (--sans-juge), le score est alors renormalisé.

score = moyenne pondérée (POIDS). On rapporte aussi coût (tokens) et latence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .latex import compiler, compilateur, document, verifier_statique
from .juge import juger_fiche
from .retriever import Index, normaliser

POIDS = {"latex": 0.15, "couverture": 0.25, "formules": 0.10, "citations": 0.10, "retrieval": 0.10, "fidelite": 0.30}
ENVS_ENONCES = ("definition", "defn", "theoreme", "theorem", "proposition", "prop", "lemme", "lemma",
                "corollaire", "corollary", "methode", "method")

# --- métriques déterministes ------------------------------------------------------------

def norm_latex(s: str) -> str:
    s = re.sub(r"\\(left|right|big|Big|bigl|bigr|displaystyle|,|;|!|quad|qquad)", "", s)
    s = re.sub(r"\\(mathrm|operatorname|text|mathbf)\{([^}]*)\}", r"\2", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return re.sub(r"[\s{}$]", "", s)


def m_latex(corps: str, compiler_pdf: bool = True) -> Dict:
    if compiler_pdf and compilateur():
        ok, journal = compiler(document(corps, "test"))
        return {"score": float(ok), "methode": compilateur(), "detail": "" if ok else journal[-400:]}
    erreurs = verifier_statique(corps)
    return {"score": float(not erreurs), "methode": "statique", "detail": "; ".join(erreurs)}


def m_couverture(corps: str, notions: List[List[str]]) -> Dict:
    texte = normaliser(corps)
    manquantes = [n[0] for n in notions if not any(normaliser(s) in texte for s in n)]
    score = 1 - len(manquantes) / len(notions) if notions else 1.0
    return {"score": score, "manquantes": manquantes}


def m_formules(corps: str, formules: List[str]) -> Dict:
    texte = norm_latex(corps)
    manquantes = [f for f in formules if norm_latex(f) not in texte]
    return {"score": 1 - len(manquantes) / len(formules) if formules else 1.0, "manquantes": manquantes}


def m_citations(corps: str, chunks_vus: List[str]) -> Dict:
    cites = re.findall(r"\\src\{([^}]*)\}", corps)
    ids = [i.strip() for c in cites for i in c.split(",")]
    valides = sum(i in chunks_vus for i in ids) / len(ids) if ids else 0.0
    # Un énoncé est « cité » si un \src apparaît dans son environnement ou juste après.
    blocs = re.findall(r"\\begin\{(%s)\}(.*?)\\end\{\1\}(?=(.{0,40}))" % "|".join(ENVS_ENONCES), corps, re.S)
    cites_env = sum("\\src{" in b[1] + b[2] for b in blocs) / len(blocs) if blocs else 0.0
    return {"score": 0.5 * valides + 0.5 * cites_env, "valides": valides, "enonces_cites": cites_env, "n": len(ids)}


def m_retrieval(chunks_vus: List[str], index: Index, sections: List[str]) -> Dict:
    lues = {normaliser(index.par_id[i].section) for i in chunks_vus if i in index.par_id}
    trouvees = [s for s in sections if any(normaliser(s) in l for l in lues)]
    return {"score": len(trouvees) / len(sections) if sections else 1.0,
            "manquantes": [s for s in sections if s not in trouvees]}


# --- LLM-juge -----------------------------------------------------------------------------

def m_fidelite(corps: str, chunks_vus: List[str], index: Index, cli, modele: str, paralleles: int = 4) -> Dict:
    return juger_fiche(corps, chunks_vus, index, cli, modele, paralleles)


# --- boucle d'évaluation --------------------------------------------------------------------

def evaluer_cas(cas: Dict, res, index: Index, cli=None, modele_juge: Optional[str] = None,
                compiler_pdf: bool = True, paralleles: int = 4) -> Dict:
    m = {
        "latex": m_latex(res.corps, compiler_pdf),
        "couverture": m_couverture(res.corps, cas.get("notions", [])),
        "formules": m_formules(res.corps, cas.get("formules", [])),
        "citations": m_citations(res.corps, res.chunks_vus),
        "retrieval": m_retrieval(res.chunks_vus, index, cas.get("sections_attendues", [])),
    }
    if modele_juge and cli is not None:
        m["fidelite"] = m_fidelite(res.corps, res.chunks_vus, index, cli, modele_juge, paralleles)
    poids = {k: POIDS[k] for k in m}
    m["score"] = sum(poids[k] * m[k]["score"] for k in m) / sum(poids.values())
    m["cout"] = {"tokens_entree": res.tokens_entree, "tokens_sortie": res.tokens_sortie,
                 "etapes": res.etapes, "duree_s": round(res.duree_s, 1), "requetes": res.requetes}
    return m


def charger_cas(chemin: Path) -> List[Dict]:
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def tableau(resultats: Dict[str, List[Dict]]) -> str:
    """resultats = {mode: [métriques par cas]} -> tableau Markdown des moyennes."""
    cles = list(POIDS) + ["score"]
    lignes = ["| mode | " + " | ".join(cles) + " | erreurs | sans erreur | tokens | s |",
              "|" + "---|" * (len(cles) + 5)]
    for mode, liste in resultats.items():
        moy = []
        for k in cles:
            vals = [(r[k] if k == "score" else r[k]["score"]) for r in liste if k in r]
            moy.append(f"{sum(vals) / len(vals):.2f}" if vals else "–")
        jugees = [r["fidelite"].get("n_erreurs", 0) for r in liste if "fidelite" in r]
        err = f"{sum(jugees) / len(jugees):.1f}" if jugees else "–"
        sans = f"{sum(e == 0 for e in jugees) / len(jugees):.0%}" if jugees else "–"
        tok = sum(r["cout"]["tokens_entree"] + r["cout"]["tokens_sortie"] for r in liste) / len(liste)
        dur = sum(r["cout"]["duree_s"] for r in liste) / len(liste)
        lignes.append(f"| {mode} | " + " | ".join(moy) + f" | {err} | {sans} | {tok:.0f} | {dur:.1f} |")
    return "\n".join(lignes)
