"""Écriture de fichiers météo EPW (EnergyPlus), pour UWG ou toute simulation bâtiment.

Entrée : un tableau horaire indexé en UTC. Le fichier EPW est en heure locale
**normale** (sans heure d'été) : la ligne « heure h » couvre l'intervalle qui se
termine à h. Une mesure instantanée à t UTC va donc sur la ligne qui se termine
à t + décalage, et un cumul horaire qui se termine à t UTC aussi.

Colonnes attendues (unités) :
    T (°C), TD (°C), RH (%), P (Pa), GHI (Wh/m², cumul de l'heure écoulée),
    WS (m/s), WD (°), N (couverture nuageuse, dixièmes) ;
    optionnelles : DNI, DHI (Wh/m²), IR (W/m²). Si absentes, DNI/DHI viennent de
    la décomposition d'Erbs et IR de la formule d'EnergyPlus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIGMA = 5.6697e-8
SOLAR_CONSTANT = 1367.0
MAX_GAP_H = 6  # trous comblés par interpolation linéaire jusqu'à 6 h, au-delà par voisinage


# --------------------------------------------------------------------------- #
# Soleil et rayonnement
# --------------------------------------------------------------------------- #
def solar_cos_zenith(times_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Cosinus de l'angle zénithal (formules NOAA, précision ~0,5°)."""
    doy = times_utc.dayofyear.to_numpy()
    hour = times_utc.hour.to_numpy() + times_utc.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_min = hour * 60 + eqtime + 4 * lon
    ha = np.radians(true_solar_min / 4 - 180)
    phi = np.radians(lat)
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)


def erbs_split(ghi: np.ndarray, times_end_utc: pd.DatetimeIndex, lat: float, lon: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Direct normal et diffus horizontal (Wh/m²) à partir du global horaire (Erbs 1982).

    Le soleil est pris au milieu de l'heure (cumul qui se termine à `times_end_utc`).
    """
    mid = times_end_utc - pd.Timedelta(minutes=30)
    cosz = solar_cos_zenith(mid, lat, lon)
    doy = mid.dayofyear.to_numpy()
    i0h = SOLAR_CONSTANT * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) * np.clip(cosz, 0, None)
    ghi = np.clip(np.nan_to_num(ghi, nan=0.0), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(i0h > 1, np.clip(ghi / i0h, 0, 1), 0.0)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3
                           + 12.336 * kt ** 4, 0.165))
    dhi = kd * ghi
    sun_up = cosz > 0.087  # soleil à plus de 5° au-dessus de l'horizon
    dni = np.where(sun_up, (ghi - dhi) / np.where(sun_up, cosz, 1), 0.0)
    dhi = np.where(sun_up, dhi, ghi)
    dni = np.clip(dni, 0, SOLAR_CONSTANT * 1.1)
    return dni, dhi


def sky_infrared(t_c: np.ndarray, td_c: np.ndarray, n_tenths: np.ndarray) -> np.ndarray:
    """Rayonnement infrarouge du ciel (W/m²), formule utilisée par EnergyPlus."""
    tdk = np.asarray(td_c) + 273.15
    n = np.clip(np.asarray(n_tenths, dtype=float), 0, 10)
    eps = (0.787 + 0.764 * np.log(tdk / 273.0)) * (1 + 0.0224 * n - 0.0035 * n ** 2 + 0.00028 * n ** 3)
    return eps * SIGMA * (np.asarray(t_c) + 273.15) ** 4


def rh_from_t_td(t_c, td_c):
    """Humidité relative (%) par la formule de Magnus."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))


def station_pressure(pmer_hpa, alt_m: float):
    """Pression au niveau de la station (Pa) depuis la pression mer (hPa)."""
    return np.asarray(pmer_hpa) * 100 * (1 - 0.0065 * alt_m / 288.15) ** 5.255


# --------------------------------------------------------------------------- #
# Écriture
# --------------------------------------------------------------------------- #
def _fill(s: pd.Series) -> pd.Series:
    return s.interpolate(limit=MAX_GAP_H, limit_direction="both").ffill().bfill()


def build_year(df_utc: pd.DataFrame, year: int, tz_hours: float, lat: float, lon: float,
               alt_m: float) -> tuple[pd.DataFrame, dict]:
    """Aligne les données sur les 8 760 lignes EPW de l'année (non bissextile).

    Renvoie (tableau EPW, part des valeurs comblées par variable).
    """
    if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366:
        raise ValueError("UWG suppose 8 760 heures : choisir une année non bissextile.")
    lst_end = pd.date_range(f"{year}-01-01 01:00", periods=8760, freq="h")
    utc = lst_end - pd.Timedelta(hours=tz_hours)
    src = df_utc.copy()
    src.index = pd.DatetimeIndex(src.index).tz_localize(None) if src.index.tz is not None else src.index
    src = src[~src.index.duplicated()].reindex(utc)

    filled = {}
    out = pd.DataFrame(index=utc)
    for col, default in (("T", None), ("TD", None), ("WS", 2.0), ("WD", 0.0), ("N", 5.0), ("GHI", 0.0)):
        s = src[col] if col in src else pd.Series(np.nan, index=utc)
        filled[col] = float(s.isna().mean())
        s = _fill(s) if s.notna().any() else s
        out[col] = s.fillna(default) if default is not None else s
    if out["T"].isna().any() or out["TD"].isna().any():
        raise ValueError("T et TD sont indispensables pour écrire un EPW.")
    out["TD"] = np.minimum(out["TD"], out["T"])
    rh = src["RH"] if "RH" in src else pd.Series(np.nan, index=utc)
    filled["RH"] = float(rh.isna().mean())
    out["RH"] = _fill(rh).fillna(pd.Series(rh_from_t_td(out["T"], out["TD"]), index=utc)).clip(1, 100)
    p = src["P"] if "P" in src else pd.Series(np.nan, index=utc)
    filled["P"] = float(p.isna().mean())
    p_std = float(station_pressure(1013.25, alt_m))
    out["P"] = _fill(p).fillna(p_std) if p.notna().any() else p_std
    out["N"] = out["N"].clip(0, 10).round()
    out["GHI"] = out["GHI"].clip(lower=0)
    # rayonnement nul la nuit (soleil sous l'horizon sur toute l'heure)
    cosz_mid = solar_cos_zenith(utc - pd.Timedelta(minutes=30), lat, lon)
    out.loc[cosz_mid < -0.05, "GHI"] = 0.0
    if "DNI" in src and "DHI" in src and src["DNI"].notna().mean() > 0.9:
        out["DNI"], out["DHI"] = _fill(src["DNI"]).clip(lower=0), _fill(src["DHI"]).clip(lower=0)
    else:
        out["DNI"], out["DHI"] = erbs_split(out["GHI"].to_numpy(), utc, lat, lon)
    if "IR" in src and src["IR"].notna().mean() > 0.9:
        out["IR"] = _fill(src["IR"])
    else:
        out["IR"] = sky_infrared(out["T"].to_numpy(), out["TD"].to_numpy(), out["N"].to_numpy())
    out["lst_end"] = lst_end
    return out, filled


def write(df_utc: pd.DataFrame, path: Path, *, year: int, tz_hours: float, lat: float, lon: float,
          alt_m: float, city: str, source: str, wmo: str = "999999") -> dict:
    """Écrit l'EPW de l'année `year` ; renvoie la part de valeurs comblées par variable."""
    out, filled = build_year(df_utc, year, tz_hours, lat, lon, alt_m)

    # Températures du sol (3 profondeurs, exigées par UWG) : moyenne mensuelle de l'air,
    # amortie et déphasée avec la profondeur.
    monthly = out.groupby(out["lst_end"].dt.month)["T"].mean().reindex(range(1, 13)).interpolate()
    mean = monthly.mean()
    ground = []
    for depth, damp, lag in ((0.5, 0.8, 1), (2.0, 0.5, 2), (4.0, 0.3, 3)):
        temps = [mean + damp * (monthly[(m - lag - 1) % 12 + 1] - mean) for m in range(1, 13)]
        ground.append(f"{depth},,,," + ",".join(f"{t:.2f}" for t in temps))
    first_day = pd.Timestamp(year=year, month=1, day=1).day_name()

    header = [
        f"LOCATION,{city},,FRA,{source},{wmo},{lat:.4f},{lon:.4f},{tz_hours:.1f},{alt_m:.1f}",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,3," + ",".join(ground),
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        f"COMMENTS 1,Genere par Helio (core/epw.py) depuis {source}",
        "COMMENTS 2,DNI/DHI par decomposition d'Erbs si absents ; IR par la formule EnergyPlus",
        f"DATA PERIODS,1,1,Data,{first_day},1/1,12/31",
    ]
    lines = []
    for r in out.itertuples():
        e = r.lst_end
        day = e - pd.Timedelta(hours=1)  # l'heure 24 appartient à la veille
        hour = 24 if e.hour == 0 else e.hour
        lines.append(",".join(str(v) for v in (
            year, day.month, day.day, hour, 60, "?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9",
            f"{r.T:.1f}", f"{r.TD:.1f}", f"{r.RH:.0f}", f"{r.P:.0f}", 9999, 9999, f"{r.IR:.0f}",
            f"{r.GHI:.0f}", f"{r.DNI:.0f}", f"{r.DHI:.0f}", 999999, 999999, 999999, 9999,
            f"{r.WD:.0f}", f"{r.WS:.1f}", f"{r.N:.0f}", f"{r.N:.0f}", 9999, 99999, 9, 999999999,
            999, 0.999, 999, 99, 999, 0, 0)))
    Path(path).write_text("\n".join(header + lines) + "\n")
    return filled


def from_meteofrance(obs: pd.DataFrame) -> pd.DataFrame:
    """Observations horaires Météo-France d'une station (index UTC) -> colonnes EPW.

    GLO en J/cm² sur l'heure -> Wh/m² (× 10 000 / 3 600) ; N en octas -> dixièmes.
    """
    out = pd.DataFrame(index=obs.index)
    out["T"], out["TD"], out["RH"] = obs.get("T"), obs.get("TD"), obs.get("U")
    out["WS"], out["WD"] = obs.get("FF"), obs.get("DD")
    out["N"] = obs["N"] * 10 / 8 if "N" in obs else np.nan
    out["GHI"] = obs["GLO"] * 10000 / 3600 if "GLO" in obs else np.nan
    if "PMER" in obs and "ALTI" in obs:
        out["P"] = station_pressure(obs["PMER"], float(obs["ALTI"].iloc[0]))
    return out
