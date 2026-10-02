"""Étape 2 : modèle PyTorch – découpage train / valid / test, entraînement, évaluation.

Idée : ERA5 donne la météo « de fond » sur une maille de ~9 km. Le modèle apprend la
correction locale  y = T_station − era5_T  à partir de :
  - la fenêtre des `fenetre` dernières heures d'ERA5 au point (chaleur accumulée, vent,
    nuages : c'est ce qui fait les nuits chaudes), lue par un GRU (ou aplatie pour un MLP) ;
  - les variables fixes du site (relief, mer, LCZ, bâti, satellite).
Prévision = era5_T + correction prédite.

Découpage (le plus honnête : stations jamais vues ET période future) :
  - les stations sont tirées au hasard en trois groupes : train / valid / test ;
  - si `date_test` est donnée : train et valid avant cette date, test après ;
  - `part_stations_test = 0` : test = toutes les stations après `date_test` (test temporel seul).
Valid sert à l'arrêt anticipé et au choix des hyperparamètres ; test ne sert qu'à la fin.

Usage :
    python modele.py                         # entraîne avec CONFIG, évalue, sauvegarde
ou depuis un notebook pour régler les hyperparamètres :
    import modele
    res = modele.entrainer({**modele.CONFIG, "lr": 3e-4, "fenetre": 48})
    res["scores_valid"]
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import config

CONFIG = {
    "departements": config.DEPARTEMENTS,     # jeux donnees/jeu/jeu_<dep>.parquet à combiner
    # --- variables -----------------------------------------------------------------------
    # dynamiques : lues heure par heure sur la fenêtre (heure_sin/cos et WD_sin/cos sont calculées ici)
    "dynamiques": ["era5_T", "era5_TD", "era5_RH", "era5_P", "era5_WS", "era5_WD_sin", "era5_WD_cos",
                   "era5_GHI", "era5_IR", "era5_RR", "era5_TS", "cos_zenith", "heure_sin", "heure_cos"],
    # fixes : toutes les colonnes numériques de variables fixes, sauf celles-ci
    "fixes_exclues": ["lat", "lon", "lst_dates"],
    # --- découpage -----------------------------------------------------------------------
    "part_stations_test": 0.2,
    "part_stations_valid": 0.15,
    "stations_test": None,                   # ou liste explicite de NUM_POSTE (remplace la part)
    "date_test": "2025-10-01",               # None : pas de coupure temporelle
    "graine": 0,
    # --- modèle ----------------------------------------------------------------------------
    "type": "gru",                           # "gru" ou "mlp" (fenêtre aplatie)
    "fenetre": 24,                           # heures d'ERA5 vues par le modèle (1 = heure courante seule)
    "gru_taille": 64,
    "gru_couches": 1,
    "mlp_tailles": [128, 64],                # couches cachées de la tête (après le GRU)
    "dropout": 0.1,
    # --- entraînement ----------------------------------------------------------------------
    "perte": "mse",                          # "mse" ou "huber" (moins sensible aux mesures aberrantes)
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 1024,
    "epoques": 30,
    "patience": 5,                           # arrêt si valid ne s'améliore plus pendant 5 époques
    "echantillons_par_epoque": None,         # ex. 200_000 pour des essais rapides ; None = tout
    "appareil": "cpu",                       # "cpu", "mps" (GPU Mac), "cuda" ou "auto"
    "nom": "helio",                          # fichier donnees/modeles/<nom>.pt
}
SEUIL_CHAUD = 30.0


# =========================================================================================
# Données
# =========================================================================================
def charger(departements: list) -> pd.DataFrame:
    """Jeux des départements mis bout à bout, triés par station puis par heure."""
    dossier = config.DONNEES / "jeu"
    presents = [d for d in departements if (dossier / f"jeu_{d}.parquet").exists()]
    for d in sorted(set(departements) - set(presents)):
        print(f"jeu absent pour le {d} (python construire_jeu.py --dep {d}) : ignoré")
    if not presents:
        raise FileNotFoundError("aucun jeu d'entraînement : lancer d'abord construire_jeu.py")
    df = pd.concat([pd.read_parquet(dossier / f"jeu_{d}.parquet") for d in presents], ignore_index=True)
    df = df.sort_values(["station", "time"]).reset_index(drop=True)
    # variables dérivées : angles -> sinus / cosinus (23 h et 0 h sont voisines, 359° et 1° aussi)
    df["heure_sin"] = np.sin(2 * np.pi * df["heure_locale"] / 24).astype("float32")
    df["heure_cos"] = np.cos(2 * np.pi * df["heure_locale"] / 24).astype("float32")
    if "era5_WD" in df:
        df["era5_WD_sin"] = np.sin(np.radians(df["era5_WD"])).astype("float32")
        df["era5_WD_cos"] = np.cos(np.radians(df["era5_WD"])).astype("float32")
    pas = df.groupby("station")["time"].diff().dropna()
    if (pas != pd.Timedelta(hours=1)).any():
        raise ValueError("heures manquantes dans une station : reconstruire le jeu (construire_jeu.py)")
    return df


def colonnes_fixes(df: pd.DataFrame, cfg: dict) -> list:
    meta = {"station", "nom", "departement", "time", "T_station", "heure_locale", "jour_annee"}
    cols = [c for c in df.columns
            if c not in meta and c not in cfg["fixes_exclues"] and c not in cfg["dynamiques"]
            and not c.startswith("era5_") and c not in ("heure_sin", "heure_cos", "cos_zenith")
            and pd.api.types.is_numeric_dtype(df[c])]
    par_station = df.groupby("station")[cols].first()
    return [c for c in cols if par_station[c].nunique(dropna=True) > 1]    # retire les constantes


def decouper(df: pd.DataFrame, cfg: dict) -> dict:
    """Masques booléens (une valeur par ligne) train / valid / test + listes de stations."""
    rng = np.random.default_rng(cfg["graine"])
    toutes = [str(s) for s in rng.permutation(sorted(df["station"].unique()))]
    n = len(toutes)
    if cfg["stations_test"]:
        test = [s for s in toutes if s in set(cfg["stations_test"])]
    elif cfg["part_stations_test"] > 0:
        test = toutes[:max(1, round(cfg["part_stations_test"] * n))]
    else:
        test = []
    reste = [s for s in toutes if s not in test]
    valid = reste[:max(1, round(cfg["part_stations_valid"] * n))]
    train = [s for s in reste if s not in valid]
    if not test:                              # test temporel seul : toutes les stations, après la date
        if not cfg["date_test"]:
            raise ValueError("ni stations de test ni date_test : rien pour tester")
        test = toutes
    st, t = df["station"], df["time"]
    avant = t < pd.Timestamp(cfg["date_test"]) if cfg["date_test"] else pd.Series(True, index=df.index)
    return {"train": (st.isin(train) & avant).to_numpy(), "valid": (st.isin(valid) & avant).to_numpy(),
            "test": (st.isin(test) & ~avant).to_numpy() if cfg["date_test"] else st.isin(test).to_numpy(),
            "stations": {"train": train, "valid": valid, "test": test}}


class Donnees:
    """Tableaux numpy prêts pour PyTorch, normalisés avec les statistiques du train uniquement.

    X_dyn [lignes, variables dynamiques], X_fix [stations, variables fixes], y [lignes].
    Un échantillon = une ligne i ; sa fenêtre = les lignes i-fenetre+1 … i de la même station.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict, masques: dict, normalisation: dict | None = None):
        self.dyn = [c for c in cfg["dynamiques"] if c in df]
        self.fix = colonnes_fixes(df, cfg) if normalisation is None else normalisation["fixes"]
        codes, self.stations = pd.factorize(df["station"])
        self.station_ligne = codes                                         # [lignes]
        fixes = df.groupby(codes)[self.fix].first().to_numpy("float32")    # [stations, fixes]
        if normalisation is None:                                          # statistiques du train
            tr = masques["train"]
            sta_tr = np.unique(codes[tr])
            with np.errstate(all="ignore"):
                normalisation = {"fixes": self.fix, "dyn": self.dyn,
                                 "dyn_moy": df.loc[tr, self.dyn].mean().to_numpy("float32"),
                                 "dyn_et": df.loc[tr, self.dyn].std().to_numpy("float32") + 1e-6,
                                 "fix_moy": np.nanmean(fixes[sta_tr], axis=0),
                                 "fix_et": np.nanstd(fixes[sta_tr], axis=0) + 1e-6,
                                 "fix_manque": np.isnan(fixes).any(axis=0)}
        self.norm = normalisation
        dyn = (df[self.dyn].to_numpy("float32") - normalisation["dyn_moy"]) / normalisation["dyn_et"]
        self.X_dyn = np.nan_to_num(dyn, nan=0.0).astype("float32")
        with np.errstate(all="ignore"):
            fx = (fixes - normalisation["fix_moy"]) / normalisation["fix_et"]
        manque = np.isnan(fixes[:, normalisation["fix_manque"]]).astype("float32")   # 1 = source absente
        self.X_fix = np.concatenate([np.nan_to_num(fx, nan=0.0), manque], axis=1).astype("float32")
        self.era5_T = df["era5_T"].to_numpy("float32")
        self.T = df["T_station"].to_numpy("float32")
        self.y = self.T - self.era5_T
        # première ligne de chaque station : la fenêtre ne doit pas déborder sur la station précédente
        premiere = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        rang = np.arange(len(df)) - np.repeat(premiere, np.diff(np.r_[premiere, len(df)]))
        self.utilisable = np.isfinite(self.y) & (rang >= cfg["fenetre"] - 1)
        self.fenetre = cfg["fenetre"]
        self.df = df

    def indices(self, masque: np.ndarray) -> np.ndarray:
        return np.flatnonzero(masque & self.utilisable)

    def lot(self, idx: np.ndarray, appareil):
        """(fenêtres [B, fenetre, dyn], fixes [B, fix], y [B]) en tenseurs."""
        fen = idx[:, None] + np.arange(-self.fenetre + 1, 1)
        return (torch.from_numpy(self.X_dyn[fen]).to(appareil),
                torch.from_numpy(self.X_fix[self.station_ligne[idx]]).to(appareil),
                torch.from_numpy(self.y[idx]).to(appareil))


# =========================================================================================
# Modèle
# =========================================================================================
class ModeleHelio(nn.Module):
    def __init__(self, n_dyn: int, n_fix: int, cfg: dict):
        super().__init__()
        if cfg["type"] == "gru":
            self.gru = nn.GRU(n_dyn, cfg["gru_taille"], num_layers=cfg["gru_couches"], batch_first=True,
                              dropout=cfg["dropout"] if cfg["gru_couches"] > 1 else 0.0)
            n = cfg["gru_taille"] + n_fix
        else:
            self.gru = None
            n = n_dyn * cfg["fenetre"] + n_fix
        couches = []
        for h in cfg["mlp_tailles"]:
            couches += [nn.Linear(n, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            n = h
        couches.append(nn.Linear(n, 1))
        self.tete = nn.Sequential(*couches)

    def forward(self, x_dyn: torch.Tensor, x_fix: torch.Tensor) -> torch.Tensor:
        if self.gru is not None:
            _, h = self.gru(x_dyn)          # h : [couches, B, taille] -> état final de la dernière couche
            z = h[-1]
        else:
            z = x_dyn.flatten(1)
        return self.tete(torch.cat([z, x_fix], dim=1)).squeeze(1)


def _appareil(nom: str) -> torch.device:
    if nom == "auto":
        nom = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(nom)


@torch.no_grad()
def predire_correction(modele: ModeleHelio, d: Donnees, idx: np.ndarray, appareil, batch: int = 8192) -> np.ndarray:
    modele.eval()
    sorties = [modele(*d.lot(idx[i:i + batch], appareil)[:2]).cpu().numpy() for i in range(0, len(idx), batch)]
    return np.concatenate(sorties) if sorties else np.array([], dtype="float32")


# =========================================================================================
# Scores
# =========================================================================================
def scores(d: Donnees, idx: np.ndarray, correction: np.ndarray, biais_train: float) -> pd.DataFrame:
    """RMSE / MAE / biais (prévu − mesuré, °C) : ERA5 brut, ERA5 + biais moyen, modèle.

    Périmètres : toutes les heures, nuit (21 h - 6 h locales), heures ≥ 30 °C,
    Tmin et Tmax journalières (jours avec au moins 20 heures mesurées).
    """
    sub = d.df.iloc[idx]
    obs, era5 = d.T[idx], d.era5_T[idx]
    previsions = {"ERA5 brut": era5, "ERA5 + biais moyen": era5 + biais_train, "modèle": era5 + correction}
    decalage = pd.to_timedelta(sub["departement"].map(config.decalage_utc), unit="h")
    jour = (sub["time"] + decalage).dt.floor("D").to_numpy()          # jour en heure locale
    nuit = sub["heure_locale"].isin([21, 22, 23, 0, 1, 2, 3, 4, 5, 6]).to_numpy()
    lignes = []
    for nom, p in previsions.items():
        e = p - obs
        tab = pd.DataFrame({"p": p, "o": obs, "s": sub["station"].to_numpy(), "j": jour})
        n_heures = tab.groupby(["s", "j"])["o"].transform("size")
        gj = tab[n_heures >= 20].groupby(["s", "j"])
        for perimetre, err in (("toutes les heures", e), ("nuit (21 h - 6 h)", e[nuit]),
                               (f"heures ≥ {SEUIL_CHAUD:.0f} °C", e[obs >= SEUIL_CHAUD]),
                               ("Tmin journalière", (gj["p"].min() - gj["o"].min()).to_numpy()),
                               ("Tmax journalière", (gj["p"].max() - gj["o"].max()).to_numpy())):
            lignes.append({"prévision": nom, "périmètre": perimetre, "n": len(err),
                           "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                           "MAE": float(np.mean(np.abs(err))) if len(err) else np.nan,
                           "biais": float(np.mean(err)) if len(err) else np.nan})
    return pd.DataFrame(lignes)


def tableau_rmse(s: pd.DataFrame) -> pd.DataFrame:
    """Vue compacte : une ligne par périmètre, une colonne par prévision (RMSE °C)."""
    t = s.pivot(index="périmètre", columns="prévision", values="RMSE")
    return t.loc[s["périmètre"].unique(), ["ERA5 brut", "ERA5 + biais moyen", "modèle"]].round(2)


# =========================================================================================
# Entraînement
# =========================================================================================
def entrainer(cfg: dict = CONFIG, df: pd.DataFrame | None = None, sauver: bool = True) -> dict:
    """Entraîne, arrête au meilleur score valid, évalue sur valid et test. Renvoie un dict de résultats."""
    torch.manual_seed(cfg["graine"])
    rng = np.random.default_rng(cfg["graine"])
    appareil = _appareil(cfg["appareil"])
    df = charger(cfg["departements"]) if df is None else df
    m = decouper(df, cfg)
    d = Donnees(df, cfg, m)
    idx = {k: d.indices(m[k]) for k in ("train", "valid", "test")}
    print(f"Stations train / valid / test : {len(m['stations']['train'])} / {len(m['stations']['valid'])} / "
          f"{len(m['stations']['test'])} ; échantillons : {len(idx['train']):,} / {len(idx['valid']):,} / "
          f"{len(idx['test']):,} ; {len(d.dyn)} variables dynamiques x {cfg['fenetre']} h, "
          f"{d.X_fix.shape[1]} fixes ; appareil {appareil}")
    if not len(idx["train"]) or not len(idx["valid"]):
        raise ValueError("train ou valid vide : vérifier date_test et les parts de stations")

    modele = ModeleHelio(len(d.dyn), d.X_fix.shape[1], cfg).to(appareil)
    opt = torch.optim.AdamW(modele.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    planif = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    perte = nn.HuberLoss(delta=1.0) if cfg["perte"] == "huber" else nn.MSELoss()
    meilleur, etat, attente, historique = np.inf, None, 0, []

    for ep in range(1, cfg["epoques"] + 1):
        t0 = time.time()
        modele.train()
        ordre = rng.permutation(idx["train"])
        if cfg["echantillons_par_epoque"]:
            ordre = ordre[:cfg["echantillons_par_epoque"]]
        somme, n = 0.0, 0
        for i in range(0, len(ordre), cfg["batch"]):
            xd, xf, y = d.lot(ordre[i:i + cfg["batch"]], appareil)
            opt.zero_grad()
            loss = perte(modele(xd, xf), y)
            loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            opt.step()
            somme += loss.item() * len(y)
            n += len(y)
        corr = predire_correction(modele, d, idx["valid"], appareil)
        rmse_valid = float(np.sqrt(np.mean((corr - d.y[idx["valid"]]) ** 2)))
        planif.step(rmse_valid)
        historique.append({"epoque": ep, "perte_train": somme / n, "rmse_valid": rmse_valid,
                           "lr": opt.param_groups[0]["lr"]})
        print(f"  époque {ep:3d}  perte train {somme / n:.3f}  RMSE valid {rmse_valid:.3f} °C  "
              f"lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t0:.0f} s)")
        if rmse_valid < meilleur - 1e-4:
            meilleur, attente = rmse_valid, 0
            etat = {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= cfg["patience"]:
                print("  arrêt anticipé")
                break
    modele.load_state_dict(etat)

    biais = float(np.nanmean(d.y[idx["train"]]))
    res = {"config": cfg, "historique": pd.DataFrame(historique), "stations": m["stations"],
           "modele": modele, "donnees": d}
    for k in ("valid", "test"):
        if len(idx[k]):
            res[f"scores_{k}"] = scores(d, idx[k], predire_correction(modele, d, idx[k], appareil), biais)
    if sauver:
        chemin = config.dossier("modeles") / f"{cfg['nom']}.pt"
        torch.save({"etat": etat, "config": cfg, "normalisation": d.norm, "stations": m["stations"],
                    "biais_train": biais}, chemin)
        res["chemin"] = chemin
    return res


def charger_modele(nom: str = CONFIG["nom"]):
    """(modèle, sauvegarde) : la sauvegarde contient config, normalisation et découpage."""
    s = torch.load(config.DONNEES / "modeles" / f"{nom}.pt", weights_only=False)
    n_fix = len(s["normalisation"]["fixes"]) + int(np.sum(s["normalisation"]["fix_manque"]))
    modele = ModeleHelio(len(s["normalisation"]["dyn"]), n_fix, s["config"])
    modele.load_state_dict(s["etat"])
    return modele.eval(), s


if __name__ == "__main__":
    r = entrainer(CONFIG)
    for k in ("valid", "test"):
        if f"scores_{k}" in r:
            print(f"\n=== {k.upper()} – RMSE (°C), erreur = prévu − mesuré ===")
            print(tableau_rmse(r[f"scores_{k}"]).to_string())
    print("\nModèle enregistré :", r.get("chemin"))
