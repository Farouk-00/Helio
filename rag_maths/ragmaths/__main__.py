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
    titre = a.titre or (a.demande if len(a.demande) <= 60 else "Fiche de révision")
    tex = document(res.corps, titre)
    sortie = Path(a.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(tex, encoding="utf-8")
    ok, journal = compiler(tex, sortie.with_suffix(".pdf"))
    lignes = journal.strip().splitlines()
    pdf = "oui" if ok else f"non ({lignes[-1] if lignes else '?'})"
    print(f"{sortie} ({res.etapes} étape(s), {len(res.chunks_vus)} chunks lus, "
          f"{res.tokens_entree}+{res.tokens_sortie} tokens, {res.duree_s:.1f} s) ; PDF : {pdf}")
    print("Outils appelés : " + " | ".join(res.requetes))


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
            m = evaluer_cas(c, res, agent.index, cli, juge, paralleles=CONFIG.juge_paralleles)
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


def _fiche(chemin: str):
    from .latex import extraire_corps
    from .llm import client
    from .retriever import Index

    if not CONFIG.fichier_index.exists():
        sys.exit("Index absent : lancer d'abord `python -m ragmaths indexer`.")
    return extraire_corps(Path(chemin).read_text(encoding="utf-8")), Index.charger(CONFIG.fichier_index), client()


def _sauver(nom: str, data) -> Path:
    dossier = RACINE / "eval" / "resultats"
    dossier.mkdir(parents=True, exist_ok=True)
    f = dossier / f"{time.strftime('%Y%m%d-%H%M%S')}_{nom}.json"
    f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return f


def cmd_juger(a) -> None:
    from .juge import juger_fiche

    corps, index, cli = _fiche(a.fiche)
    m = juger_fiche(corps, [], index, cli, CONFIG.modele_juge, CONFIG.juge_paralleles)
    print(f"{m['n']} énoncés : {m['n_erreurs']} erreur(s), {m.get('n_non_soutenus', 0)} non soutenu(s) ; "
          f"score {m['score']:.2f}")
    for e in m.get("erreurs", []):
        print(f"\n✗ {e['titre'] or '(sans titre)'}\n  fiche : {e['probleme']}\n  cours : {e['preuve']}")
    for e in m.get("non_soutenus", []):
        print(f"\n? {e['titre'] or '(sans titre)'} ({e['verdict']}) {e['probleme'] or ''}")
    print(f"\nDétail : {_sauver('juge_' + Path(a.fiche).stem, m)}")


def cmd_tester_juge(a) -> None:
    from .juge import tableau_test_juge, tester_juge

    corps, index, cli = _fiche(a.fiche)
    r = tester_juge(corps, index, cli, CONFIG.modele_juge, a.n, a.graine, CONFIG.juge_paralleles)
    print(f"Juge : {CONFIG.modele_juge} ; {r['n_enonces']} énoncés dans la fiche\n")
    print(tableau_test_juge(r))
    for nom, d in r["par_type"].items():
        for x in d["details"]:
            if x["verdict"] != "ERREUR":
                print(f"\nratée [{nom}] {x['titre']} : {x['extrait_altere']}")
    print(f"\nDétail : {_sauver('test_juge_' + Path(a.fiche).stem, r)}")


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
    s = sp.add_parser("juger", help="vérifie une fiche énoncé par énoncé (LLM-juge)")
    s.add_argument("fiche")
    s.set_defaults(f=cmd_juger)
    s = sp.add_parser("tester-juge", help="injecte des erreurs dans une fiche et mesure le juge")
    s.add_argument("fiche")
    s.add_argument("--n", type=int, default=3, help="erreurs injectées par type")
    s.add_argument("--graine", type=int, default=0)
    s.set_defaults(f=cmd_tester_juge)
    s = sp.add_parser("servir")
    s.add_argument("--hote", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_servir)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
