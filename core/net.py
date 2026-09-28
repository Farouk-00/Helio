"""Accès réseau partagés : téléchargement en flux, JSON avec reprises."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from core.config import HTTP_TIMEOUT


def download(url: str, dest: Path, progress=None) -> Path:
    """Téléchargement en flux ; progress(fraction) est optionnel."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(done / total, 1.0))
    tmp.rename(dest)
    return dest


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             retries: int = 3) -> requests.Response:
    """GET avec quelques reprises (erreurs réseau, 429, 5xx). Renvoie la réponse."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries:
                r.raise_for_status()
            time.sleep(max(2 ** attempt, float(r.headers.get("retry-after", 0) or 0)))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("inatteignable")
