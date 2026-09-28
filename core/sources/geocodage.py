"""Géocodage d'adresses (Géoplateforme IGN, base BAN)."""
from __future__ import annotations

from core.config import GEOCODAGE_URL
from core.net import get_json


def search(query: str, limit: int = 5) -> list[dict]:
    """Adresses candidates : [{label, lat, lon, score, citycode}]."""
    if len(query.strip()) < 3:
        return []
    r = get_json(GEOCODAGE_URL, params={"q": query, "limit": limit, "index": "address"})
    out = []
    for f in r.json().get("features", []):
        lon, lat = f["geometry"]["coordinates"]
        p = f["properties"]
        out.append({"label": p.get("label", query), "lat": lat, "lon": lon,
                    "score": p.get("score"), "citycode": p.get("citycode")})
    return out
