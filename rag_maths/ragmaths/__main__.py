"""Point d'entrée : python -m ragmaths {indexer,resumer,evaluer,servir}."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .config import CONFIG, RACINE


def cmd_indexer(a) -> None:
    from .ingest import charger_cours
    from .llm import embedder
    from .retriever import Index

    chunks = charger_cours(Path(a.dossier), CONFIG.taille_chunk, CONFIG.recouvrement)
    if not chunks:
        sys.exit(f"Aucun cours (.tex/.md/.txt/.pdf) dans {a.dossier}")
    embed = embedder()
    vecs = embed([c.section + "\n" + c.texte for c in chunks]) if embed else None
    Index(chunks, vecs).sauver(CONFIG.fichier_index)
    print(f"{len(chunks)} chunks, {len({c.source for c in chunks})} fichier(s), "
          f"embeddings : {'oui' if vecs else 'non (BM25 seul)'} -> {CONFIG.fichier_index}")


def _agent():
    from .agent import Agent
    from .llm import client, embedder
    from .retriever import Index

    if not CONFIG.fichier_index.exists():
        sys.exit("Index absent : lancer d'abord `python -m ragmaths indexer`.")
    cli = client()
    return Agent(Index.charger(CONFIG.fichier_index), cli, embed=embedder(cli)), cli


def cmd_resumer(a) -> None:
    from .latex import compiler, document

    agent, _ = _agent()
    res = agent.resumer(a.demande, mode=a.mode)
    tex = document(res.corps, a.titre or a.demande)
    sortie = Path(a.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(tex, encoding="utf-8")
    ok, journal = compiler(tex, sortie.with_suffix(".pdf"))
    lignes = journal.strip().splitlines()
    pdf = "oui" if ok else f"non ({lignes[-1] if lignes else '?'})"
    print(f"{sortie} ({res.etapes} étape(s), {len(res.chunks_vus)} chunks lus, "
          f"{res.tokens_entree}+{res.tokens_sortie} tokens, {res.duree_s:.1f} s) ; PDF : {pdf}")


def cmd_evaluer(a) -> None:
    from .evaluate import charger_cas, evaluer_cas, tableau

    agent, cli = _agent()
    cas = charger_cas(Path(a.cas))
    juge = None if a.sans_juge else CONFIG.modele_juge
    resultats = {}
    for mode in a.modes:
        resultats[mode] = []
        for c in cas:
            res = agent.resumer(c["demande"], mode=mode)
            m = evaluer_cas(c, res, agent.index, cli, juge)
            m["id"] = c["id"]
            resultats[mode].append(m)
            print(f"[{mode}] {c['id']}: score {m['score']:.2f}")
    print("\n" + tableau(resultats))
    dossier = RACINE / "eval" / "resultats"
    dossier.mkdir(parents=True, exist_ok=True)
    f = dossier / f"{time.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^a-z0-9.-]', '_', agent.modele)}.json"
    f.write_text(json.dumps({"modele": agent.modele, "juge": juge, "resultats": resultats},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDétail : {f}")


def cmd_servir(a) -> None:
    import uvicorn

    uvicorn.run("ragmaths.server:app", host=a.hote, port=a.port)


def main() -> None:
    p = argparse.ArgumentParser(prog="ragmaths")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("indexer")
    s.add_argument("--dossier", default=str(CONFIG.dossier_cours))
    s.set_defaults(f=cmd_indexer)
    s = sp.add_parser("resumer")
    s.add_argument("demande")
    s.add_argument("--mode", choices=["agent", "rag"], default="agent")
    s.add_argument("--titre", default="")
    s.add_argument("--sortie", default=str(RACINE / "sorties" / "resume.tex"))
    s.set_defaults(f=cmd_resumer)
    s = sp.add_parser("evaluer")
    s.add_argument("--cas", default=str(RACINE / "eval" / "cas.jsonl"))
    s.add_argument("--modes", nargs="+", default=["rag", "agent"])
    s.add_argument("--sans-juge", action="store_true")
    s.set_defaults(f=cmd_evaluer)
    s = sp.add_parser("servir")
    s.add_argument("--hote", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_servir)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
