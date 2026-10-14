#!/usr/bin/env bash
# Installation sur un VPS Debian/Ubuntu. À lancer en root depuis le dossier rag_maths copié
# sur le serveur (ex. `rsync -av rag_maths/ vps:/tmp/rag_maths/`).
set -euo pipefail

DEST=/opt/rag_maths
apt-get update -q
apt-get install -y -q python3-venv python3-pip rsync caddy
# Compilation LaTeX (facultatif mais utile pour la métrique « latex ») :
apt-get install -y -q --no-install-recommends texlive-latex-recommended texlive-lang-french

id ragmaths >/dev/null 2>&1 || useradd --system --home "$DEST" --shell /usr/sbin/nologin ragmaths
mkdir -p "$DEST"
rsync -a --exclude .venv --exclude __pycache__ ./ "$DEST"/
mkdir -p "$DEST"/data "$DEST"/sorties "$DEST"/eval/resultats
python3 -m venv "$DEST"/.venv
"$DEST"/.venv/bin/pip install -q -r "$DEST"/requirements.txt
[ -f "$DEST"/.env ] || { cp "$DEST"/.env.example "$DEST"/.env; echo ">> Remplir $DEST/.env puis relancer."; exit 1; }
chmod 600 "$DEST"/.env
chown -R ragmaths: "$DEST"

sudo -u ragmaths bash -c "cd $DEST && set -a && . ./.env && .venv/bin/python -m ragmaths indexer"

cp "$DEST"/deploy/ragmaths.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ragmaths
systemctl --no-pager status ragmaths | head -5

# HTTPS automatique : remplacer par votre domaine dans deploy/Caddyfile.
cp "$DEST"/deploy/Caddyfile /etc/caddy/Caddyfile && systemctl reload caddy

# Évaluation de non-régression chaque nuit (résultats dans eval/resultats/).
echo "0 3 * * * ragmaths cd $DEST && set -a && . ./.env && .venv/bin/python -m ragmaths evaluer >> $DEST/eval/resultats/cron.log 2>&1" \
  > /etc/cron.d/ragmaths-eval
echo "OK : curl -s localhost:8000/health"
