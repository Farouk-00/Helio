"""Mise en forme et vérification du LaTeX produit."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

PREAMBULE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage[margin=2cm]{geometry}
\theoremstyle{definition}
\newtheorem{definition}{Définition}
\theoremstyle{plain}
\newtheorem{theoreme}{Théorème}
\newtheorem{proposition}{Proposition}
\newtheorem{lemme}{Lemme}
\newtheorem{corollaire}{Corollaire}
\theoremstyle{remark}
\newtheorem*{remarque}{Remarque}
\newtheorem*{exemple}{Exemple}
\newtheorem*{methode}{Méthode}
% Alias anglais : le modèle les emploie souvent quand le cours est en anglais.
\theoremstyle{definition}
\newtheorem{defn}[definition]{Définition}
\theoremstyle{plain}
\newtheorem{theorem}[theoreme]{Théorème}
\newtheorem{lemma}[lemme]{Lemme}
\newtheorem{corollary}[corollaire]{Corollaire}
\newtheorem{prop}[proposition]{Proposition}
\theoremstyle{remark}
\newtheorem*{remark}{Remarque}
\newtheorem*{example}{Exemple}
\newtheorem*{method}{Méthode}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Prob}{\mathbb{P}}
\newcommand{\src}[1]{\textsuperscript{\tiny[#1]}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\K}{\mathbb{K}}
\newcommand{\N}{\mathbb{N}}
"""

ENVS_AUTORISES = "definition, theoreme, proposition, lemme, corollaire, remarque, exemple, methode"


def extraire_corps(reponse: str) -> str:
    """Récupère le corps LaTeX (le modèle met parfois des ``` ou un préambule)."""
    m = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", reponse, re.S)
    corps = m.group(1) if m else reponse
    if "\\begin{document}" in corps:
        corps = corps.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    return corps.strip()


def document(corps: str, titre: str) -> str:
    return (
        PREAMBULE
        + f"\\title{{{titre}}}\n\\date{{}}\n\\begin{{document}}\n\\maketitle\n\n"
        + corps
        + "\n\n\\end{document}\n"
    )


def verifier_statique(corps: str) -> List[str]:
    """Contrôles sans compilateur : accolades, \\begin/\\end, $ et \\[ \\] équilibrés."""
    erreurs = []
    sans_echap = re.sub(r"\\[{}$]", "", corps)
    sans_echap = re.sub(r"(?<!\\)%.*", "", sans_echap)  # commentaires
    if sans_echap.count("{") != sans_echap.count("}"):
        erreurs.append("accolades déséquilibrées")
    pile = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", sans_echap):
        if m.group(1) == "begin":
            pile.append(m.group(2))
        elif not pile or pile.pop() != m.group(2):
            erreurs.append(f"\\end{{{m.group(2)}}} sans \\begin correspondant")
    erreurs += [f"\\begin{{{e}}} non fermé" for e in pile]
    if len(re.findall(r"(?<!\$)\$(?!\$)", sans_echap)) % 2:
        erreurs.append("nombre impair de $")
    if sans_echap.count("\\[") != sans_echap.count("\\]"):
        erreurs.append("\\[ et \\] déséquilibrés")
    return erreurs


def compilateur() -> str:
    for nom in ("tectonic", "pdflatex"):
        if shutil.which(nom):
            return nom
    return ""


def compiler(tex: str, sortie_pdf: Path = None) -> Tuple[bool, str]:
    """Compile si tectonic ou pdflatex est installé. Renvoie (ok, journal)."""
    outil = compilateur()
    if not outil:
        return False, "aucun compilateur"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "resume.tex"
        f.write_text(tex, encoding="utf-8")
        cmd = (
            ["tectonic", "--chatter", "minimal", str(f)]
            if outil == "tectonic"
            else ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f.name]
        )
        try:
            p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = p.returncode == 0 and (Path(d) / "resume.pdf").exists()
        if ok and sortie_pdf:
            sortie_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(d) / "resume.pdf", sortie_pdf)
        return ok, (p.stdout + p.stderr)[-2000:]
