"""Lecture des cours (.tex, .md, .txt, .pdf) et découpage en chunks.

Le découpage suit la structure du cours (sections LaTeX / titres Markdown / pages PDF),
puis regroupe les paragraphes jusqu'à ~`taille` caractères. On coupe sur les lignes vides
pour ne jamais casser un environnement `align` ou un `$$ ... $$` au milieu.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

RE_SECTION_TEX = re.compile(r"^\\(chapter|section|subsection)\*?\{(.+?)\}", re.M)
RE_TITRE_MD = re.compile(r"^(#{1,3})\s+(.+)$", re.M)


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
    from pypdf import PdfReader  # import paresseux : dépendance facultative

    lecteur = PdfReader(str(chemin))
    return [(f"page {i + 1}", p.extract_text() or "") for i, p in enumerate(lecteur.pages)]


def _regrouper(texte: str, taille: int, recouvrement: int) -> Iterable[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", texte) if p.strip()]
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
