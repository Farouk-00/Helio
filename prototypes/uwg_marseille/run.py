"""Prototype UWG – Marseille, été 2025.

Question : UWG (Urban Weather Generator, MIT), forcé par la station « rurale » de
Marignane (aéroport), retrouve-t-il la température mesurée en ville à
Marseille-Observatoire (Palais Longchamp) ?

Étapes :
1. EPW 2025 de Marignane depuis les observations horaires Météo-France (core/epw.py).
2. Morphologie dans un rayon de 500 m autour de Marseille-Observatoire : BDNB
   (hauteur, emprise, surface de façades, usages) ; végétation : NDVI Sentinel-2
   (scène d'été la moins nuageuse, Planetary Computer).
3. UWG du 25 mai au 31 août 2025 (7 jours de mise en route écartés).
4. Comparaison horaire juin-août avec la mesure de Marseille-Observatoire, contre
   la référence naïve « T de Marignane ».

Pré-requis : fichier Stations du département 13 contenant 2025 (onglet Stations,
période « latest »), accès réseau à l'API BDNB et à Planetary Computer.

Lancement, depuis la racine du dépôt :
    python -m prototypes.uwg_marseille.run
Sorties : data/prototypes/uwg_marseille/ (EPW, CSV horaire, métriques JSON, figures HTML).
"""
from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import pystac_client
import rasterio
from plotly.subplots import make_subplots
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point

from core import epw
from core.config import DATA_DIR, PC_STAC_URL
from core.sources import bdnb
from core.sources import meteofrance_stations as mfs

OUT = DATA_DIR / "prototypes" / "uwg_marseille"
YEAR = 2025
TZ = 1  # heure normale de la France métropolitaine (UTC+1), convention EPW
REF = "13054001"      # MARIGNANE (aéroport) : forçage « rural »
CIBLE = "13055001"    # MARSEILLE-OBS (Palais Longchamp) : vérité en ville
RAYON = 500           # m, zone de morphologie autour de la cible
SPINUP = ("2025-05-25", 7)      # début de simulation, jours écartés
EVAL = ("2025-06-01", "2025-08-31")


# --------------------------------------------------------------------------- #
# 1. Observations et EPW
# --------------------------------------------------------------------------- #
def observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = mfs.load("13")
    if df.empty:
        sys.exit("Aucune donnée Stations pour le 13 : télécharge la période « latest » dans l'onglet Stations.")
    sta = {}
    for code in (REF, CIBLE):
        s = df[df["NUM_POSTE"] == code].set_index("time").sort_index()
        s = s[s.index.year == YEAR]
        if len(s) < 8000:
            sys.exit(f"Station {code} : seulement {len(s)} heures en {YEAR} (il faut l'année complète).")
        sta[code] = s
    return sta[REF], sta[CIBLE]


def build_epw(ref: pd.DataFrame) -> tuple[str, dict]:
    obs = ref.copy()
    for v in ("T", "TD", "U", "FF", "DD", "PMER", "N", "GLO"):  # garder aussi les valeurs en code 9 (non validées)
        if v in obs:
            obs[v] = obs[v].where(obs[f"Q{v}"].isin([0, 1, 9]) if f"Q{v}" in obs else obs[v].notna())
    path = OUT / f"marignane_{YEAR}.epw"
    filled = epw.write(epw.from_meteofrance(obs), path, year=YEAR, tz_hours=TZ,
                       lat=float(ref["LAT"].iloc[0]), lon=float(ref["LON"].iloc[0]),
                       alt_m=float(ref["ALTI"].iloc[0]), city="Marignane", source=f"Meteo-France {REF}")
    return str(path), filled


# --------------------------------------------------------------------------- #
# 2. Morphologie (BDNB) et végétation (Sentinel-2)
# --------------------------------------------------------------------------- #
def morphology(lat: float, lon: float) -> dict:
    if not bdnb.is_ready(lat, lon, RAYON):
        print(f"BDNB : téléchargement des bâtiments dans {RAYON} m (quelques minutes)…")
        bdnb.fetch(lat, lon, RAYON)
    b = bdnb.load(lat, lon, RAYON)
    site = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(bdnb.CRS).iloc[0]
    disk = site.buffer(RAYON)
    b = b[b.geometry.centroid.within(disk)].copy()     # bâtiments dont le centre est dans le disque
    b["emprise"] = b.geometry.area
    b["perimetre"] = b.geometry.length
    h = b["hauteur_mean"].astype(float)
    b["h"] = h.fillna(h.median())
    usage = b.groupby(b["usage_principal_bdnb_open"].fillna("inconnu"))["emprise"].sum()
    return {
        "n_batiments": int(len(b)),
        "bldheight": float((b["h"] * b["emprise"]).sum() / b["emprise"].sum()),
        "blddensity": float(b["emprise"].sum() / disk.area),
        "vertohor": float((b["perimetre"] * b["h"]).sum() / disk.area),
        "part_hauteur_manquante": float(h.isna().mean()),
        "usages_emprise": (usage / usage.sum()).round(3).to_dict(),
    }


def vegetation(lat: float, lon: float, res: float = 10.0) -> dict:
    """Parts d'arbres (NDVI >= 0,5) et d'herbe (0,3 <= NDVI < 0,5) dans le disque."""
    client = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    d = 0.01
    items = list(client.search(collections=["sentinel-2-l2a"], bbox=[lon - d, lat - d, lon + d, lat + d],
                               datetime=f"{EVAL[0]}/{EVAL[1]}").items())
    it = min(items, key=lambda i: i.properties["eo:cloud_cover"])
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    crs = pt.estimate_utm_crs()
    p = pt.to_crs(crs).iloc[0]
    n = int(np.ceil(RAYON / res))
    size = 2 * n + 1
    transform = from_origin(p.x - (n + 0.5) * res, p.y + (n + 0.5) * res, res, res)

    def read(asset: str) -> np.ndarray:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(it.assets[asset].href) as src, \
                WarpedVRT(src, crs=crs, transform=transform, width=size, height=size,
                          resampling=Resampling.nearest, nodata=0) as vrt:
            return vrt.read(1).astype("float32")

    # Depuis la version de traitement 04.00 (janv. 2022), réflectance = (CN - 1000) / 10 000
    offset = 1000.0 if float(it.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    red, nir, scl = read("B04") - offset, read("B08") - offset, read("SCL")
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    disk = (xx ** 2 + yy ** 2) * res ** 2 <= RAYON ** 2
    ok = disk & np.isin(scl, [2, 4, 5, 6, 7]) & (red + nir > 0)
    ndvi = np.where(ok, (nir - red) / np.where(red + nir > 0, red + nir, 1), np.nan)
    v = ndvi[ok]
    return {"scene": it.id, "nuages_scene": it.properties["eo:cloud_cover"],
            "part_pixels_valides": float(ok.sum() / disk.sum()),
            "treecover": float((v >= 0.5).mean()), "grasscover": float(((v >= 0.3) & (v < 0.5)).mean()),
            "ndvi_median": float(np.median(v))}


def building_mix(usages: dict) -> list:
    """Parc de bâtiments UWG (références DOE, construites avant 1980) d'après les usages BDNB."""
    res = sum(v for k, v in usages.items() if k.startswith("Résidentiel"))
    ter = usages.get("Tertiaire", 0.0)
    autre = max(0.0, 1 - res - ter)
    mix = [("midriseapartment", "pre80", res), ("medoffice", "pre80", ter), ("standaloneretail", "pre80", autre)]
    mix = [(t, e, round(f, 3)) for t, e, f in mix if f >= 0.01]
    tot = sum(f for *_, f in mix)
    mix = [(t, e, f / tot) for t, e, f in mix]
    mix[-1] = (mix[-1][0], mix[-1][1], 1 - sum(f for *_, f in mix[:-1]))  # somme exactement 1
    return mix


# --------------------------------------------------------------------------- #
# 3. UWG
# --------------------------------------------------------------------------- #
def run_uwg(epw_path: str, params: dict) -> tuple[pd.Series, int]:
    """Lance UWG ; en cas de divergence numérique, relance avec un pas de temps plus fin.

    Renvoie (T urbaine horaire indexée en UTC, pas de temps utilisé en s).
    """
    from uwg import UWG

    start = pd.Timestamp(SPINUP[0])
    ndays = (pd.Timestamp(EVAL[1]) - start).days + 1
    for dtsim in (300, 150, 60):
        model = UWG.from_param_args(
            epw_path=epw_path, new_epw_dir=str(OUT), new_epw_name=f"marseille_obs_uwg_{YEAR}.epw",
            bldheight=params["bldheight"], blddensity=params["blddensity"], vertohor=params["vertohor"],
            grasscover=params["grasscover"], treecover=params["treecover"], zone="3A",
            month=start.month, day=start.day, nday=ndays, dtsim=dtsim, bld=params["bld"])
        model.generate()
        try:
            model.simulate()
        except Exception as exc:  # UWG lève Exception("FATAL ERROR! …") quand il diverge
            print(f"UWG a divergé avec un pas de {dtsim} s ({str(exc).splitlines()[0]}), nouvel essai…")
            continue
        model.write_epw()
        rows = pd.read_csv(model.new_epw_path, skiprows=8, header=None)
        utc = pd.date_range(f"{YEAR}-01-01 01:00", periods=8760, freq="h") - pd.Timedelta(hours=TZ)
        return pd.Series(rows[6].to_numpy(dtype=float), index=utc, name="T_uwg"), dtsim
    sys.exit("UWG diverge même avec un pas de 60 s.")


# --------------------------------------------------------------------------- #
# 4. Évaluation
# --------------------------------------------------------------------------- #
def scores(err: pd.Series) -> dict:
    return {"biais": round(float(err.mean()), 2), "MAE": round(float(err.abs().mean()), 2),
            "RMSE": round(float(np.sqrt((err ** 2).mean())), 2), "n": int(err.notna().sum())}


def evaluate(ref: pd.DataFrame, cible: pd.DataFrame, t_uwg: pd.Series, alt_diff: float) -> tuple[pd.DataFrame, dict]:
    t_obs = cible["T"].where(cible["QT"].isin([0, 1]))
    t_ref = ref["T"].where(ref["QT"].isin([0, 1]))
    h = pd.DataFrame({"T_obs": t_obs, "T_rural": t_ref, "T_uwg": t_uwg}).loc[EVAL[0]:f"{EVAL[1]} 23:00"]
    h["T_uwg_alt"] = h["T_uwg"] - 0.0065 * alt_diff   # variante : gradient standard pour l'altitude
    h = h.dropna()
    local = h.index + pd.Timedelta(hours=2)          # heure légale d'été, pour la lecture
    nuit = (local.hour >= 22) | (local.hour < 6)
    m = {}
    for name, col in (("référence naïve (T Marignane)", "T_rural"), ("UWG", "T_uwg"),
                      ("UWG + correction d'altitude", "T_uwg_alt")):
        e = h[col] - h["T_obs"]
        daily = h.groupby(local.date).agg(["min", "max"])
        m[name] = {"horaire": scores(e), "nuit (22h-6h)": scores(e[nuit]), "jour": scores(e[~nuit]),
                   "Tmin journalière": scores(daily[(col, "min")] - daily[("T_obs", "min")]),
                   "Tmax journalière": scores(daily[(col, "max")] - daily[("T_obs", "max")])}
    # décalage horaire éventuel de la sortie UWG (diagnostic, pas un réglage)
    m["diagnostic décalage UWG (RMSE selon décalage en h)"] = {
        str(k): round(float(np.sqrt(((h["T_uwg"].shift(k) - h["T_obs"]) ** 2).mean())), 3) for k in (-2, -1, 0, 1, 2)}
    return h, m


def figures(h: pd.DataFrame) -> None:
    local = h.index + pd.Timedelta(hours=2)
    prof = pd.DataFrame({"mesuré (Marseille-Obs − Marignane)": h["T_obs"] - h["T_rural"],
                         "UWG (UWG − Marignane)": h["T_uwg"] - h["T_rural"]}).groupby(local.hour).mean()
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Îlot de chaleur moyen selon l'heure (heure légale), juin-août 2025",
        "Températures horaires (°C)"), vertical_spacing=0.12)
    for c in prof:
        fig.add_trace(go.Scatter(x=prof.index, y=prof[c], name=c, mode="lines+markers"), row=1, col=1)
    for c, n in (("T_obs", "Marseille-Obs (mesure)"), ("T_uwg", "UWG"), ("T_rural", "Marignane (forçage)")):
        fig.add_trace(go.Scatter(x=local, y=h[c], name=n, mode="lines", line=dict(width=1)), row=2, col=1)
    fig.update_layout(height=850, template="plotly_white")
    fig.update_yaxes(title_text="écart (°C)", row=1, col=1)
    fig.write_html(OUT / "resultats.html", include_plotlyjs="cdn")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref, cible = observations()
    lat, lon = float(cible["LAT"].iloc[0]), float(cible["LON"].iloc[0])
    alt_diff = float(cible["ALTI"].iloc[0]) - float(ref["ALTI"].iloc[0])

    epw_path, filled = build_epw(ref)
    print("EPW Marignane écrit :", epw_path, "| part comblée :", {k: round(v, 3) for k, v in filled.items()})
    morph = morphology(lat, lon)
    veg = vegetation(lat, lon)
    params = {**{k: morph[k] for k in ("bldheight", "blddensity", "vertohor")},
              "treecover": veg["treecover"], "grasscover": veg["grasscover"],
              "bld": building_mix(morph["usages_emprise"])}
    exces = params["blddensity"] + params["treecover"] + params["grasscover"] - 1
    if exces > 0:  # contrainte UWG : bâti + herbe + arbres <= 1
        params["grasscover"] = max(0.0, params["grasscover"] - exces)
    print("Paramètres UWG :", json.dumps(params, ensure_ascii=False, default=str))

    t_uwg, dtsim = run_uwg(epw_path, params)
    params["dtsim_s"] = dtsim
    h, m = evaluate(ref, cible, t_uwg, alt_diff)
    h.to_csv(OUT / "horaire.csv")
    figures(h)
    report = {"forçage": f"{REF} Marignane", "cible": f"{CIBLE} Marseille-Observatoire",
              "écart d'altitude cible - forçage (m)": alt_diff, "epw_part_comblee": filled,
              "morphologie": morph, "vegetation": veg, "parametres_uwg": params, "scores": m}
    (OUT / "metriques.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("Sorties :", OUT)


if __name__ == "__main__":
    main()
