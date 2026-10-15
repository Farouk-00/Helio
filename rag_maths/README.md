# rag-maths : agent RAG qui résume des cours de maths en LaTeX

Mistral (API Mistral directe ou via OpenRouter, au choix dans `.env`) + recherche hybride BM25 / `mistral-embed` + boucle d'appels d'outils
+ sortie LaTeX compilable + jeu d'évaluation avec score composite. Python ≥ 3.9, ~600 lignes.

```
cours/*.tex|md|pdf ──indexer──> data/index.json (chunks par section + embeddings)
                                       │
demande ──> Agent (Mistral, tools: plan_du_cours, lire_section, chercher_cours) ──> corps LaTeX + \src{cID}
                                       │
                         latex.document() ──> .tex / .pdf     evaluate ──> score [0,1]
```

## Démarrage (local)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # FOURNISSEUR=mistral + MISTRAL_API_KEY (ou OpenRouter)
# mettre ses cours dans cours/ (un exemple est fourni : réduction des endomorphismes)
python -m ragmaths indexer
python -m ragmaths resumer "Fiche de révision sur la diagonalisation" --sortie sorties/diag.tex
python -m ragmaths evaluer        # compare les modes rag et agent sur eval/cas.jsonl
python -m pytest -q tests         # tests hors ligne (faux client)
```

## Choix de conception (à défendre en entretien)
- **Découpage par structure** (`\section`, titres Markdown, pages PDF) puis paragraphes : on ne
  coupe jamais un énoncé ou un `align` en deux. Chaque chunk garde `source > section`.
- **Recherche hybride** : BM25 maison (le vocabulaire mathématique est très discriminant,
  `\det`, « scindé ») + embeddings `mistral-embed`, fusion RRF. Sans embeddings, ça marche quand même.
- **Agent vs RAG simple** : en mode `agent`, Mistral appelle `plan_du_cours`, puis
  `lire_section` (chapitre entier, dans l'ordre : retrieval *par structure*) et/ou
  `chercher_cours` (notion précise : retrieval *par similarité*) ; en mode `rag`, une seule
  recherche. Au dernier tour (`ETAPES_MAX`, 8 par défaut) on force `tool_choice="none"` et une
  consigne de rédaction ; si le modèle écrit malgré tout un appel d'outil en JSON, une relance.
- **LaTeX robuste** : le modèle n'écrit que le corps, avec une liste fermée d'environnements ;
  le préambule est fixé par le code → beaucoup moins d'échecs de compilation.
- **Traçabilité** : chaque énoncé porte `\src{c12}` → vérifiable, et mesurable.
- **Refus hors corpus** : `% ABSENT DU COURS` plutôt qu'inventer (cas `hors_cours` du jeu d'éval).

## Métrique d'évaluation (`ragmaths/evaluate.py`)
Score composite dans [0, 1] par cas, moyenné sur `eval/cas.jsonl` (cas annotés à la main) :

| composante | poids | mesure | type |
|---|---|---|---|
| latex | 0,15 | le document compile (tectonic/pdflatex), sinon contrôle statique | déterministe |
| couverture | 0,25 | part des notions attendues présentes (avec synonymes) | déterministe |
| formules | 0,10 | formules clés retrouvées après normalisation LaTeX | déterministe |
| citations | 0,10 | ½ citations pointant vers un chunk réellement lu + ½ énoncés cités | déterministe |
| retrieval | 0,10 | rappel des sections attendues parmi les chunks lus | déterministe |
| fidélité | 0,30 | LLM-juge (autre modèle Mistral) : part des affirmations soutenues par les extraits ; une hypothèse oubliée = contredite | LLM |

Plus coût (tokens) et latence par mode, et le détail JSON (notions manquantes, affirmations
contredites) dans `eval/resultats/`. `--sans-juge` pour une éval gratuite et reproductible.

Limites connues (et pistes) : couverture par chaînes (→ juge pour l'équivalence sémantique),
juge à valider contre des notes humaines sur ~20 fiches (corrélation de Spearman), variance du
générateur (→ 3 tirages par cas), petit jeu de cas (→ l'enrichir à chaque bug observé).

## Déploiement sur VPS
```bash
rsync -av --exclude .venv ./ root@vps:/tmp/rag_maths/
ssh root@vps 'cd /tmp/rag_maths && bash deploy/installer_vps.sh'
curl -H "Authorization: Bearer $API_TOKEN" -X POST https://rag.mon-domaine.fr/resume \
     -H 'Content-Type: application/json' -d '{"demande": "Fiche sur Cayley-Hamilton"}'
```
systemd (`deploy/ragmaths.service`, redémarrage auto), Caddy (HTTPS automatique), API
FastAPI derrière jeton, éval de non-régression chaque nuit par cron.
