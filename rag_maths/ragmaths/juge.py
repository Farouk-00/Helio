"""LLM-juge de fidélité, et test du juge par injection d'erreurs.

1. On découpe la fiche en ÉNONCÉS (environnements definition, theoreme, remarque…).
2. Chaque énoncé est vérifié SEUL contre les SEULS chunks qu'il cite (contexte court =
   jugement plus précis), par un appel au juge qui doit comparer les hypothèses une à une
   et citer mot à mot le passage fautif.
3. Indicateur principal : le NOMBRE D'ERREURS (objectif 0) ; le score est la part
   d'énoncés corrects (une erreur pèse 1/n énoncés, pas 1/n affirmations).

`tester_juge` mesure le juge lui-même : on altère un énoncé (hypothèse supprimée,
constante changée, inégalité inversée, sens inversé) et on regarde s'il le détecte
(rappel), puis on lui soumet les énoncés intacts (taux de faux positifs).
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .llm import appeler, texte
from .retriever import Index, tokeniser

# Environnements contenant des énoncés vérifiables (noms français + alias anglais).
ENVS_JUGES = (
    "definition", "defn", "theoreme", "theorem", "proposition", "prop", "lemme", "lemma",
    "corollaire", "corollary", "remarque", "remark", "methode", "method",
)
RE_ENONCE = re.compile(
    # (?=…) : on regarde les 40 caractères suivants (un \\src juste après \\end) sans les
    # consommer, sinon ils avaleraient le \\begin de l'énoncé suivant.
    r"\\begin\{(%s)\}(\[[^\]]*\])?(.*?)\\end\{\1\}(?=(.{0,40}))" % "|".join(ENVS_JUGES), re.S
)
RE_SRC = re.compile(r"\\src\{([^}]*)\}")

JUGE_ENONCE = """Tu es un correcteur de mathématiques très rigoureux. On te donne un ÉNONCÉ \
tiré d'une fiche de révision et les EXTRAITS du cours dont il est censé venir.
Vérifie-le mot à mot :
1. Liste les hypothèses de ce résultat dans le cours, puis celles présentes dans l'énoncé.
2. Compare formules, constantes, signes, inégalités, sens des comparaisons \
(plus grand / plus petit, minimise / maximise, croît / décroît) et quantificateurs.
Verdict :
- ERREUR si une hypothèse du cours manque dans l'énoncé, ou si une formule, une constante, \
une inégalité ou une affirmation diffère du cours ou le contredit ;
- NON_SOUTENU si les extraits ne permettent pas de vérifier l'énoncé ;
- CORRECT sinon (une reformulation équivalente ou des notations différentes sont correctes).
Réponds UNIQUEMENT en JSON, dans cet ordre :
{"hypotheses_cours": ["..."], "hypotheses_enonce": ["..."], \
"probleme": "passage fautif CITÉ MOT À MOT depuis l'énoncé (vide si CORRECT)", \
"preuve": "passage des extraits qui le contredit ou le soutient", \
"verdict": "CORRECT|ERREUR|NON_SOUTENU"}"""


@dataclass
class Enonce:
    env: str
    titre: str
    texte: str  # environnement complet, tel qu'écrit dans la fiche
    ids: List[str] = field(default_factory=list)


def extraire_enonces(corps: str) -> List[Enonce]:
    res = []
    for m in RE_ENONCE.finditer(corps):
        ids = [i.strip() for g in RE_SRC.findall(m.group(3) + m.group(4)) for i in g.split(",") if i.strip()]
        res.append(Enonce(m.group(1), (m.group(2) or "").strip("[]"), m.group(0), ids))
    return res


def _contexte(e: Enonce, index: Index, chunks_vus: List[str], max_chunks: int = 6) -> List[str]:
    """Chunks cités par l'énoncé ; à défaut, les chunks lus les plus proches (BM25)."""
    cites = [i for i in e.ids if i in index.par_id]
    if cites:
        return cites[:max_chunks]
    scores = index.bm25.scores(tokeniser(e.texte))
    rang = {c.id: s for c, s in zip(index.chunks, scores)}
    candidats = [i for i in chunks_vus if i in index.par_id] or list(index.par_id)
    return sorted(candidats, key=lambda i: -rang.get(i, 0))[:max_chunks]


def juger_enonce(texte_enonce: str, ids: List[str], index: Index, cli, modele: str) -> Dict:
    extraits = "\n\n".join(f"[{i}] ({index.par_id[i].section})\n{index.par_id[i].texte}" for i in ids)
    try:
        r = appeler(
            cli.chat.completions.create,
            model=modele,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUGE_ENONCE},
                {"role": "user", "content": f"EXTRAITS :\n{extraits}\n\nÉNONCÉ :\n{texte_enonce}"},
            ],
        )
    except Exception as e:  # un énoncé non jugé ne doit pas faire perdre tout le run
        return {"verdict": "INVALIDE", "probleme": f"erreur API : {type(e).__name__}: {e}"[:300]}
    brut = texte(r.choices[0].message.content)
    m = re.search(r"\{.*\}", brut, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        v = {}
    verdict = str(v.get("verdict", "")).upper().replace(" ", "_")
    v["verdict"] = verdict if verdict in ("CORRECT", "ERREUR", "NON_SOUTENU") else "INVALIDE"
    return v


def _paralleliser(fn: Callable, elements: List, n: int) -> List:
    if n <= 1 or len(elements) <= 1:
        return [fn(x) for x in elements]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, elements))


def juger_fiche(corps: str, chunks_vus: List[str], index: Index, cli, modele: str,
                paralleles: int = 4) -> Dict:
    enonces = extraire_enonces(corps)
    if not enonces:
        return {"score": 0.0, "n": 0, "n_erreurs": 0, "detail": "aucun énoncé (environnement) dans la fiche"}

    def un(e: Enonce) -> Dict:
        v = juger_enonce(e.texte, _contexte(e, index, chunks_vus), index, cli, modele)
        return {"env": e.env, "titre": e.titre, "ids": e.ids, **v}

    verdicts = _paralleliser(un, enonces, paralleles)
    n = len(verdicts)
    erreurs = [v for v in verdicts if v["verdict"] == "ERREUR"]
    non_soutenus = [v for v in verdicts if v["verdict"] in ("NON_SOUTENU", "INVALIDE")]
    return {
        "score": sum(v["verdict"] == "CORRECT" for v in verdicts) / n,
        "n": n,
        "n_erreurs": len(erreurs),
        "n_non_soutenus": len(non_soutenus),
        "erreurs": [{k: v.get(k) for k in ("titre", "probleme", "preuve", "hypotheses_cours", "hypotheses_enonce")}
                    for v in erreurs],
        "non_soutenus": [{k: v.get(k) for k in ("titre", "probleme", "verdict")} for v in non_soutenus],
    }


# --- injection d'erreurs (test du juge) ------------------------------------------------------

RE_MATHS = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$]+\$", re.S)

RE_HYPOTHESES = [
    # « If X is injective (one-to-one), then: » / « Si ..., alors »
    re.compile(r"\b(?:If|Si|When|Lorsque|Quand)\s+[^.:;]*?,\s*(?:then|alors)?\s*"),
    # « Assume ... . » / « On suppose ... . »
    re.compile(r"\b(?:Assume|Suppose|On suppose|Supposons)\b[^.]*\.\s*"),
    # « Under the Gaussian assumption ..., » / « Sous l'hypothèse ..., »
    re.compile(r"\b(?:Under\s+(?:the\s+)?[^,]*?assumptions?|Sous l'hypothèse[^,]*?)[^,]*,\s*"),
]

SENS = [
    ("smaller", "larger"), ("larger", "smaller"), ("increases", "decreases"), ("decreases", "increases"),
    ("minimizes", "maximizes"), ("maximizes", "minimizes"), ("minimize", "maximize"), ("maximize", "minimize"),
    ("independent", "dependent"), ("unbiased", "biased"), ("at least", "at most"), ("at most", "at least"),
    ("upper", "lower"), ("lower", "upper"), ("plus petit", "plus grand"), ("plus grand", "plus petit"),
    ("croissant", "décroissant"), ("minimise", "maximise"), ("maximise", "minimise"),
    ("indépendant", "dépendant"), ("sans biais", "biaisé"), ("au moins", "au plus"), ("au plus", "au moins"),
]

INEGALITES = [("\\leq", "\\geq"), ("\\geq", "\\leq"), ("\\le", "\\ge"), ("\\ge", "\\le"),
              ("\\leqslant", "\\geqslant"), ("\\geqslant", "\\leqslant"), ("<", ">"), (">", "<")]


def _dans_maths(s: str, fn: Callable[[str], Optional[str]]) -> Optional[str]:
    """Applique fn au 1er segment mathématique qu'elle modifie ; None si aucun."""
    for m in RE_MATHS.finditer(s):
        nouveau = fn(m.group(0))
        if nouveau is not None and nouveau != m.group(0):
            return s[:m.start()] + nouveau + s[m.end():]
    return None


def muter_hypothese(s: str) -> Optional[str]:
    for motif in RE_HYPOTHESES:
        m = motif.search(s)
        if m:
            return s[:m.start()] + s[m.end():]
    return None


def muter_constante(s: str) -> Optional[str]:
    def f(maths: str) -> Optional[str]:
        m = re.search(r"(?<![\w\\^_.])([1-9])(?![\w.])", maths)
        if not m:
            return None
        d = int(m.group(1))
        return maths[:m.start(1)] + str(d + 1 if d < 9 else 8) + maths[m.end(1):]
    return _dans_maths(s, f)


def muter_inegalite(s: str) -> Optional[str]:
    def f(maths: str) -> Optional[str]:
        for a, b in INEGALITES:
            motif = re.escape(a) + (r"(?![a-zA-Z])" if a.startswith("\\") else "")
            if a in "<>":
                motif = r"(?<!\\)" + motif  # pas \langle, \rangle
            m = re.search(motif, maths)
            if m:
                return maths[:m.start()] + b + maths[m.end():]
        return None
    return _dans_maths(s, f)


def muter_sens(s: str) -> Optional[str]:
    # On ne touche pas aux commandes LaTeX ni aux maths : seulement au texte.
    hors_maths = RE_MATHS.sub(lambda m: "\0" * len(m.group(0)), s)
    for a, b in SENS:
        m = re.search(r"\b" + re.escape(a) + r"\b", hors_maths)
        if m:
            return s[:m.start()] + b + s[m.end():]
    return None


MUTATIONS: Dict[str, Callable[[str], Optional[str]]] = {
    "hypothese_supprimee": muter_hypothese,
    "constante_changee": muter_constante,
    "inegalite_inversee": muter_inegalite,
    "sens_inverse": muter_sens,
}


def preparer_mutations(enonces: List[Enonce], n_par_type: int, graine: int = 0) -> List[Tuple[str, Enonce, str]]:
    """[(type, énoncé d'origine, texte altéré)], au plus n_par_type par type."""
    hasard = random.Random(graine)
    res = []
    for nom, fn in MUTATIONS.items():
        # Retirer une phrase de contexte d'une remarque ne la rend pas forcément fausse :
        # la suppression d'hypothèse ne vise que les énoncés formels.
        cibles = [e for e in enonces if nom != "hypothese_supprimee" or e.env not in ("remarque", "remark")]
        candidats = [(e, fn(e.texte)) for e in cibles]
        candidats = [(e, t) for e, t in candidats if t]
        for e, t in hasard.sample(candidats, min(n_par_type, len(candidats))):
            res.append((nom, e, t))
    return res


def tester_juge(corps: str, index: Index, cli, modele: str, n_par_type: int = 3, graine: int = 0,
                paralleles: int = 4) -> Dict:
    enonces = extraire_enonces(corps)
    mutations = preparer_mutations(enonces, n_par_type, graine)
    originaux = list({id(e): e for _, e, _ in mutations}.values())

    def juger(t: str, e: Enonce) -> Dict:
        return juger_enonce(t, _contexte(e, index, []), index, cli, modele)

    v_mut = _paralleliser(lambda x: juger(x[2], x[1]), mutations, paralleles)
    v_orig = _paralleliser(lambda e: juger(e.texte, e), originaux, paralleles)

    par_type: Dict[str, Dict] = {}
    for (nom, e, t), v in zip(mutations, v_mut):
        d = par_type.setdefault(nom, {"injectees": 0, "detectees": 0, "non_soutenues": 0, "erreurs_api": 0,
                                      "details": []})
        if str(v.get("probleme", "")).startswith("erreur API"):
            d["erreurs_api"] += 1  # non jugé : exclu du rappel
            continue
        d["injectees"] += 1
        d["detectees"] += v["verdict"] == "ERREUR"
        d["non_soutenues"] += v["verdict"] in ("NON_SOUTENU", "INVALIDE")
        d["details"].append({"titre": e.titre, "verdict": v["verdict"], "probleme": v.get("probleme"),
                             "extrait_altere": _diff(e.texte, t)})
    for d in par_type.values():
        d["rappel"] = d["detectees"] / d["injectees"] if d["injectees"] else 0.0
    n_mut = sum(d["injectees"] for d in par_type.values())
    juges = [(e, v) for e, v in zip(originaux, v_orig) if not str(v.get("probleme", "")).startswith("erreur API")]
    faux_positifs = [{"titre": e.titre, "probleme": v.get("probleme")} for e, v in juges if v["verdict"] == "ERREUR"]
    return {
        "rappel_global": sum(d["detectees"] for d in par_type.values()) / n_mut if n_mut else None,
        "par_type": par_type,
        "faux_positifs": len(faux_positifs) / len(juges) if juges else None,
        "faux_positifs_detail": faux_positifs,
        "n_enonces": len(enonces),
    }


def _diff(avant: str, apres: str, marge: int = 40) -> str:
    """Zone modifiée, pour lire le rapport : « …avant… => …après… »."""
    i = 0
    while i < min(len(avant), len(apres)) and avant[i] == apres[i]:
        i += 1
    j = 0
    while j < min(len(avant), len(apres)) - i and avant[-1 - j] == apres[-1 - j]:
        j += 1
    a = avant[max(0, i - marge): len(avant) - j + marge]
    b = apres[max(0, i - marge): len(apres) - j + marge]
    return " ".join(a.split()) + "  =>  " + " ".join(b.split())


def tableau_test_juge(r: Dict) -> str:
    lignes = ["| erreur injectée | injectées | détectées | non soutenues | rappel |", "|---|---|---|---|---|"]
    for nom, d in r["par_type"].items():
        api = f" ({d['erreurs_api']} non jugées : erreur API)" if d["erreurs_api"] else ""
        lignes.append(f"| {nom} | {d['injectees']} | {d['detectees']} | {d['non_soutenues']} | {d['rappel']:.0%}{api} |")
    if r["rappel_global"] is not None:
        lignes.append(f"| **total** | | | | **{r['rappel_global']:.0%}** |")
    if r["faux_positifs"] is not None:
        lignes.append(f"\nFaux positifs (énoncés intacts jugés ERREUR) : {r['faux_positifs']:.0%}")
    return "\n".join(lignes)
