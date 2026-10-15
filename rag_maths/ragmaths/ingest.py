"""Lecture des cours (.tex, .md, .txt, .pdf) et découpage en chunks.

Le découpage suit la structure du cours : `\\section` en LaTeX, titres en Markdown, et pour
les PDF le sommaire intégré (signets) ou, à défaut, les titres numérotés détectés dans le
texte. Puis on regroupe les paragraphes jusqu'à ~`taille` caractères, en coupant sur les
lignes vides pour ne pas casser un environnement `align` ou un `$$ ... $$` au milieu.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

RE_SECTION_TEX = re.compile(r"^\\(chapter|section|subsection)\*?\{(.+?)\}", re.M)
RE_TITRE_MD = re.compile(r"^(#{1,3})\s+(.+)$", re.M)
# PDF sans signets : « 2.3 Sub-Gaussian variables », « Chapter 4 ... » seuls sur leur ligne.
RE_TITRE_PDF = re.compile(
    r"^[ \t]*((?:Chapter|Chapitre|CHAPTER|CHAPITRE)[ \t]+\d+[^\n]{0,80}"
    r"|\d{1,2}(?:\.\d{1,2}){0,2}\.?[ \t]+[A-ZÀ-Ý][^\n]{2,80}?)[ \t]*$",
    re.M,
)
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


@dataclass
class Chunk:
    id: str
    source: str
    section: str
    texte: str

    def to_dict(self) -> dict:
        return asdict(self)


def _blocs_tex(texte: str) -> List[Tuple[str, str]]:
    # On ignore le préambule s'il y en a un.
    if "\\begin{document}" in texte:
        texte = texte.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    return _decouper_sur(texte, RE_SECTION_TEX, groupe_titre=2)


def _blocs_md(texte: str) -> List[Tuple[str, str]]:
    return _decouper_sur(texte, RE_TITRE_MD, groupe_titre=2)


def _decouper_sur(texte: str, motif: re.Pattern, groupe_titre: int) -> List[Tuple[str, str]]:
    blocs, titre, debut = [], "(début)", 0
    for m in motif.finditer(texte):
        if texte[debut:m.start()].strip():
            blocs.append((titre, texte[debut:m.start()]))
        titre, debut = m.group(groupe_titre).strip(), m.start()
    if texte[debut:].strip():
        blocs.append((titre, texte[debut:]))
    return blocs


def _blocs_pdf(chemin: Path) -> List[Tuple[str, str]]:
    import logging

    from pypdf import PdfReader  # import paresseux : dépendance facultative

    # pypdf avertit à chaque police Type1 sans fontTools : bruyant et sans conséquence ici.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    lecteur = PdfReader(str(chemin))
    pages = [_nettoyer_page(p.extract_text() or "") for p in lecteur.pages]
    pages = _retirer_entetes(pages)
    texte, debuts = "", []
    for p in pages:
        debuts.append(len(texte))
        texte += p + "\n\n"

    entrees = _sommaire(lecteur)
    if entrees:
        return _decouper_par_sommaire(texte, debuts, entrees)
    blocs = _blocs_titres_pdf(texte)
    if len(blocs) > 1:
        return blocs
    return [(f"p. {i + 1}", p) for i, p in enumerate(pages)]  # dernier recours


def _nettoyer_page(texte: str) -> str:
    for lig, rempl in LIGATURES.items():
        texte = texte.replace(lig, rempl)
    # Mots coupés en fin de ligne (« gradi-\nents ») : on les recolle.
    return re.sub(r"(\w)-\n(\w)", r"\1\2", texte)


def _retirer_entetes(pages: List[str]) -> List[str]:
    """Supprime en-têtes et pieds de page courants (« 142 CHAPTER 5. OPTIMIZATION… »).

    Une 1re ou dernière ligne est un en-tête si, chiffres masqués, elle revient sur au moins
    3 pages en contenant un numéro, ou sur au moins 30 % des pages, ou si elle n'est qu'un numéro.
    """
    def cle(ligne: str) -> str:
        return re.sub(r"\d+", "#", ligne.strip().lower())

    def bords(p: str) -> List[str]:
        lignes = [l for l in p.split("\n") if l.strip()]
        return [lignes[0], lignes[-1]] if lignes else []

    freq = Counter(cle(l) for p in pages for l in set(bords(p)))
    res = []
    for p in pages:
        lignes = p.split("\n")
        for idx in _indices_bords(lignes):
            l = lignes[idx]
            f = freq[cle(l)]
            # Avec numéro de page : 3 occurrences suffisent ; sans : il faut qu'elle soit partout.
            if (f >= 3 and re.search(r"\d", l)) or f >= max(5, 0.3 * len(pages)) \
                    or re.fullmatch(r"\s*[\divxlcIVXLC]{1,6}\s*", l):
                lignes[idx] = ""
        res.append("\n".join(lignes))
    return res


def _indices_bords(lignes: List[str]) -> List[int]:
    pleins = [i for i, l in enumerate(lignes) if l.strip()]
    return sorted({pleins[0], pleins[-1]}) if pleins else []


def _sommaire(lecteur) -> List[Tuple[int, str, int]]:
    """Signets du PDF -> [(niveau, titre, n° de page 0-indexé)]."""
    res: List[Tuple[int, str, int]] = []

    def parcourir(noeuds, niveau: int) -> None:
        for n in noeuds:
            if isinstance(n, list):
                parcourir(n, niveau + 1)
                continue
            try:
                page = lecteur.get_destination_page_number(n)
            except Exception:
                continue
            titre = " ".join(str(getattr(n, "title", "")).split())
            if titre and page is not None and page >= 0:
                res.append((niveau, titre, page))

    try:
        parcourir(lecteur.outline, 0)
    except Exception:
        return []
    return res


def _chercher_titre(texte: str, titre: str, debut: int, fin: int) -> Tuple[int, str]:
    """Cherche le titre dans texte[debut:fin] -> (position, numéro « 1.2.1 » ou "").

    On ne compare que les lettres et chiffres (30 premiers), en tolérant n'importe quels
    espaces ou signes entre eux : l'extraction PDF ajoute des espaces, et les signets
    perdent symboles et ponctuation (« Bernstein's Inequality () », tirets, guillemets).
    """
    car = [c for c in titre if c.isalnum()][:30]
    if len(car) < 3:
        return -1, ""
    motif = r"[\W_]*".join(re.escape(c) for c in car)
    # D'abord en début de ligne (éventuellement après un numéro « 2.3 »), sinon n'importe où.
    m = re.compile(r"(?m)^[ \t]*(?:(\d+(?:\.\d+)*)\.?[ \t]*)?" + motif, re.I).search(texte, debut, fin)
    if m:
        return m.start(), m.group(1) or ""
    m = re.compile(motif, re.I).search(texte, debut, fin)
    return (m.start(), "") if m else (-1, "")


def _decouper_par_sommaire(texte: str, debuts: List[int], entrees) -> List[Tuple[str, str]]:
    trouves = []  # [niveau, titre, position, numéro]
    for niveau, titre, page in entrees:
        if page >= len(debuts):
            continue
        fin = debuts[page + 1] if page + 1 < len(debuts) else len(texte)
        pos, num = _chercher_titre(texte, titre, debuts[page], fin)
        if pos < 0:
            pos = debuts[page]  # titre introuvable : début de sa page
        trouves.append([niveau, titre, max(pos, trouves[-1][2] if trouves else 0), num])
    # Un chapitre sans numéro prend celui de sa 1re sous-section (« 1.1 » -> « 1 »).
    for e, suivant in zip(trouves, trouves[1:]):
        m = re.match(r"(\d+)\.\d+$", suivant[3])
        if not e[3] and suivant[0] == e[0] + 1 and m:
            e[3] = m.group(1)
    points: List[Tuple[int, str]] = []
    chemin: List[str] = []
    for niveau, titre, pos, num in trouves:
        chemin = chemin[:niveau] + [f"{num} {titre}" if num else titre]
        points.append((pos, " > ".join(chemin[-3:])))  # 3 derniers niveaux : assez parlant
    if not points:
        return [("(document)", texte)]
    blocs = [("(début)", texte[:points[0][0]])] if points[0][0] > 0 else []
    for (pos, nom), (suivant, _) in zip(points, points[1:] + [(len(texte), "")]):
        blocs.append((nom, texte[pos:suivant]))
    return [(nom, t) for nom, t in blocs if t.strip()]


def _ressemble_a_un_titre(ligne: str) -> bool:
    """Écarte les faux positifs : lignes de formule, phrases, sommaire (« … 313 »)."""
    lettres = sum(c.isalpha() for c in ligne)
    non_blancs = sum(not c.isspace() for c in ligne)
    return (
        not ligne.startswith("0")
        and len(ligne.split()) <= 12
        and not re.search(r"[.,;:=]$|\s\d+$|=", ligne)
        and lettres >= 0.7 * non_blancs
    )


def _blocs_titres_pdf(texte: str) -> List[Tuple[str, str]]:
    """PDF sans signets : titres numérotés courts, sans ponctuation finale."""
    bons = [m for m in RE_TITRE_PDF.finditer(texte) if _ressemble_a_un_titre(m.group(1).strip())]
    if not bons:
        return [("(document)", texte)]
    blocs = [("(début)", texte[:bons[0].start()])] if texte[:bons[0].start()].strip() else []
    for m, suivant in zip(bons, bons[1:] + [None]):
        fin = suivant.start() if suivant else len(texte)
        blocs.append((" ".join(m.group(1).split()), texte[m.start():fin]))
    return blocs


def _regrouper(texte: str, taille: int, recouvrement: int) -> Iterable[str]:
    paras = []
    for p in re.split(r"\n\s*\n", texte):
        p = p.strip()
        if len(p) <= taille:
            paras += [p] if p else []
            continue
        morceau = ""  # texte PDF : peu de lignes vides, on recoupe sur les retours à la ligne
        for ligne in p.split("\n"):
            if morceau and len(morceau) + len(ligne) > taille:
                paras.append(morceau)
                morceau = ""
            morceau = (morceau + "\n" + ligne).strip()
        if morceau:
            paras.append(morceau)
    courant = ""
    for p in paras:
        if courant and len(courant) + len(p) > taille:
            yield courant
            # Recouvrement : on garde la fin du chunk précédent (paragraphes entiers si possible).
            queue = courant[-recouvrement:]
            courant = queue[queue.find("\n") + 1:] if "\n" in queue else ""
        courant = (courant + "\n\n" + p).strip()
    if courant:
        yield courant


def charger_cours(dossier: Path, taille: int = 1200, recouvrement: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    for chemin in sorted(dossier.rglob("*")):
        ext = chemin.suffix.lower()
        if ext == ".tex":
            blocs = _blocs_tex(chemin.read_text(encoding="utf-8", errors="ignore"))
        elif ext in (".md", ".txt"):
            blocs = _blocs_md(chemin.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".pdf":
            blocs = _blocs_pdf(chemin)
        else:
            continue
        nom = str(chemin.relative_to(dossier))
        for section, contenu in blocs:
            for morceau in _regrouper(contenu, taille, recouvrement):
                chunks.append(Chunk(f"c{len(chunks)}", nom, section, morceau))
    return chunks
