"""Tests sans réseau : faux client OpenAI qui rejoue un scénario d'appels d'outils."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

from ragmaths.agent import Agent
from ragmaths.config import RACINE
from ragmaths.evaluate import charger_cas, evaluer_cas, norm_latex, tableau
from ragmaths.ingest import charger_cours
from ragmaths.latex import verifier_statique
from ragmaths.retriever import Index

INDEX = Index(charger_cours(RACINE / "cours"))


def _msg(content=None, appels=None):
    tc = [NS(id=f"t{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(appels or [])]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tc or None))],
              usage=NS(prompt_tokens=100, completion_tokens=50))


class FauxClient:
    def __init__(self, reponses):
        self.reponses, self.appels = list(reponses), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.appels.append(copy.deepcopy(kw))  # instantané : la liste de messages évolue
        return self.reponses.pop(0)


def _id(section: str) -> str:
    return next(c.id for c in INDEX.chunks if section in c.section)


def test_decoupage_et_recherche():
    assert len(INDEX.chunks) >= 6
    assert {"Diagonalisation", "Polynômes annulateurs"} <= {c.section for c in INDEX.chunks}
    top = INDEX.chercher("Cayley-Hamilton polynôme annulateur", k=2)
    assert top[0].section == "Polynômes annulateurs"
    assert INDEX.chercher("matrice symétrique orthonormée", k=1)[0].section.startswith("Réduction des matrices")


def test_boucle_agent_et_metriques():
    i_ann = _id("annulateurs")
    fiche = (
        "\\section{Polynômes annulateurs}\n"
        "\\begin{theoreme}[Cayley-Hamilton] $\\chi_u(u) = 0$. \\src{%s}\\end{theoreme}\n"
        "\\begin{definition} Le polynôme minimal $\\pi_u$ divise $\\chi_u$. \\src{%s}\\end{definition}\n"
        "$u$ diagonalisable ssi annulateur scindé à racines simples. \\src{c999}\n"
        "\\section{À retenir}\\begin{itemize}\\item ok\\end{itemize}" % (i_ann, i_ann)
    )
    cli = FauxClient([
        _msg(appels=[("plan_du_cours", {}), ("chercher_cours", {"requete": "Cayley-Hamilton polynôme minimal"})]),
        _msg(content="```latex\n" + fiche + "\n```"),
    ])
    res = Agent(INDEX, cli, modele="test").resumer("Résume les polynômes annulateurs")
    assert res.etapes == 2 and i_ann in res.chunks_vus and res.tokens_entree == 200
    assert res.corps.startswith("\\section") and "```" not in res.corps
    # Le 2e appel contient bien les résultats des outils.
    roles = [m["role"] for m in cli.appels[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]

    cas = next(c for c in charger_cas(RACINE / "eval" / "cas.jsonl") if c["id"] == "annulateurs")
    m = evaluer_cas(cas, res, INDEX, compiler_pdf=False)
    assert m["latex"]["score"] == 1.0
    assert m["couverture"]["score"] == 1.0
    assert m["formules"]["score"] == 1.0
    assert m["retrieval"]["score"] == 1.0
    # 2 citations valides sur 3 (c999 inventé) ; 2 énoncés sur 2 cités.
    assert abs(m["citations"]["valides"] - 2 / 3) < 1e-9 and m["citations"]["enonces_cites"] == 1.0
    assert "fidelite" not in m and 0 < m["score"] <= 1
    assert "| agent |" in tableau({"agent": [m]})


def test_juge():
    from ragmaths.evaluate import m_fidelite

    verdicts = {"affirmations": [{"texte": "a", "verdict": "SOUTENUE"}, {"texte": "b", "verdict": "CONTREDITE"}]}
    cli = FauxClient([_msg(content=json.dumps(verdicts))])
    m = m_fidelite("fiche", [INDEX.chunks[0].id], INDEX, cli, "juge")
    assert m["score"] == 0.5 and m["contredites"] == ["b"]


def test_mode_rag_et_dernier_tour_sans_outil():
    cli = FauxClient([_msg(content="\\section{X} ok")])
    res = Agent(INDEX, cli, modele="test").resumer("diagonalisation", mode="rag")
    assert res.etapes == 1 and res.chunks_vus and "tools" not in cli.appels[0]

    from ragmaths import agent as mod
    boucle = [_msg(appels=[("chercher_cours", {"requete": "x"})])] * (mod.CONFIG.etapes_max - 1)
    cli = FauxClient(boucle + [_msg(content="fin")])
    res = Agent(INDEX, cli, modele="test").resumer("x")
    assert res.corps == "fin" and cli.appels[-1]["tool_choice"] == "none"


def test_latex_statique_et_normalisation():
    assert verifier_statique("\\begin{theoreme} $x$ \\end{theoreme}") == []
    assert verifier_statique("\\begin{theoreme} $x \\end{lemme}")
    assert norm_latex("\\left( P D P^{-1} \\right)") == norm_latex("(PDP^{-1})")


def test_contenu_en_blocs_mistral():
    blocs = [{"type": "thinking", "thinking": [{"type": "text", "text": "je réfléchis"}]},
             {"type": "text", "text": "\\section{SGD} ok"}]
    cli = FauxClient([_msg(content=blocs)])
    res = Agent(INDEX, cli, modele="test").resumer("x")
    assert res.corps == "\\section{SGD} ok"


def test_pdf_entetes_et_sommaire():
    from ragmaths.ingest import _decouper_par_sommaire, _retirer_entetes

    pages = [f"{100 + i} CHAPTER 1. PRELIMINARIES\ncorps {i}\nsuite" for i in range(4)]
    pages[1] = "101 CHAPTER 1. PRELIMINARIES\nfin de l'intro\n1.2.1 Hoeffding’s Inequality\nIf Z1..."
    pages = _retirer_entetes(pages)
    assert all("CHAPTER" not in p for p in pages)
    assert sum("suite" in p for p in pages) == 3  # ligne répétée sans numéro : gardée
    texte, debuts = "", []
    for p in pages:
        debuts.append(len(texte))
        texte += p + "\n\n"
    entrees = [(0, "Mathematical Preliminaries", 0), (1, "Concentration Inequalities", 0),
               (2, "Hoeffding's Inequality", 1), (2, "Bernstein's Inequality ()", 3)]
    blocs = dict(_decouper_par_sommaire(texte, debuts, entrees))
    nom = "Mathematical Preliminaries > Concentration Inequalities > 1.2.1 Hoeffding's Inequality"
    assert blocs[nom].lstrip().startswith("1.2.1 Hoeffding")   # coupé au titre, pas à la page
    assert "fin de l'intro" in blocs["Mathematical Preliminaries > Concentration Inequalities"]
    assert any("Bernstein" in n for n in blocs)                # introuvable -> début de page


def test_pdf_sans_signets_titres_numerotes():
    from ragmaths.ingest import _blocs_titres_pdf

    texte = ("Intro\n1.2 Concentration Inequalities\nblabla\n0. This leads to, using x = 2\n"
             "11 From Online Learning to Bandits 313\n1.2.1 Hoeffding’s Inequality\nIf Z...")
    noms = [n for n, _ in _blocs_titres_pdf(texte)]
    assert noms == ["(début)", "1.2 Concentration Inequalities", "1.2.1 Hoeffding’s Inequality"]


def test_relance_si_appel_outil_ecrit_en_texte():
    from ragmaths import agent as mod

    boucle = [_msg(appels=[("chercher_cours", {"requete": "x"})])] * (mod.CONFIG.etapes_max - 1)
    cli = FauxClient(boucle + [_msg(content='{"requete": "R2", "k": 5}'), _msg(content="\\section{R2} ok")])
    res = Agent(INDEX, cli, modele="test").resumer("x")
    assert res.corps == "\\section{R2} ok"
    # Consigne de fin collée au dernier message d'outil (pas de « user » après « tool »).
    avant_dernier = cli.appels[-2]["messages"]
    assert avant_dernier[-1]["role"] == "tool" and mod.CONSIGNE_FIN in avant_dernier[-1]["content"]
    assert [m["role"] for m in cli.appels[-1]["messages"]][-2:] == ["assistant", "user"]


def test_lire_section():
    chunks, total = INDEX.lire_section("polynomes annulateurs")  # accents/casse ignorés
    assert total == len(chunks) >= 1 and all(c.section == "Polynômes annulateurs" for c in chunks)
    cli = FauxClient([_msg(appels=[("lire_section", {"section": "Diagonalisation"})]), _msg(content="\\section{D}")])
    res = Agent(INDEX, cli, modele="test").resumer("Résume la diagonalisation")
    assert res.requetes == ["[section] Diagonalisation"] and _id("Diagonalisation") in res.chunks_vus
