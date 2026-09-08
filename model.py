"""
Smart Sim — Modele Predictif Over 2.5
Ensemble XGBoost + LightGBM avec calibration de probabilites.
Source de donnees : Supabase (via supabase_db) + cache local.
"""

import os
import json
import time
import pickle
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Compatibilité sklearn : cv="prefit" supprimé en 1.6+, remplacé par FrozenEstimator
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False


def _make_calibrated(estimator, method: str = "isotonic"):
    """Wrapper compat pour calibration sur un modèle déjà entraîné."""
    if _HAS_FROZEN:
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(estimator, cv="prefit", method=method)

from config import MIN_CONFIDENCE, SMART_BET_THRESHOLD, LEAGUES, LAST_N_MATCHES
from features import get_model_columns, build_feature_vector, detect_smart_bet

log = logging.getLogger("SmartSim.model")

_PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Ancien fichier local (fallback uniquement) ──
HISTORICAL_DATA_FILE = _PROJECT_ROOT / "historique_data.json"


# ══════════════════════════════════════════════
# ACCES DONNEES HISTORIQUES (via supabase_db)
# ══════════════════════════════════════════════
def _load_history() -> dict[str, dict]:
    """
    Charge l'historique depuis Supabase (avec cache intelligent).
    Fallback sur l'ancien historique_data.json si Supabase indisponible.
    """
    try:
        from supabase_db import load_history
        data = load_history()
        if data:
            log.info("Historique charge via supabase_db : %d matchs", len(data))
            return data
    except ImportError:
        log.warning("supabase_db non disponible — fallback JSON local")
    except Exception as e:
        log.warning("Erreur supabase_db.load_history() : %s — fallback JSON local", e)

    # Fallback : ancien fichier JSON
    return _load_local_history_legacy()


def _save_history(history: dict[str, dict], new_matches: Optional[dict] = None):
    """
    Sauvegarde les nouveaux matchs sur Supabase + cache local.
    Si new_matches est fourni, n'ecrit que les nouveaux (optimise).
    """
    try:
        from supabase_db import append_matches, _save_local_cache
        if new_matches:
            append_matches(new_matches, history)
        else:
            _save_local_cache(history)
        return
    except ImportError:
        pass
    except Exception as e:
        log.warning("Erreur supabase_db.append_matches() : %s — fallback JSON", e)

    # Fallback : ancien systeme
    _save_local_history_legacy(history)


# ── Legacy : ancien systeme JSON (utilise seulement si supabase_db indisponible) ──
def _load_local_history_legacy() -> dict[str, dict]:
    if HISTORICAL_DATA_FILE.exists():
        try:
            with open(HISTORICAL_DATA_FILE, "r") as f:
                data = json.load(f)
            log.info("Historique JSON legacy charge : %d matchs.", len(data))
            return data
        except (json.JSONDecodeError, Exception) as e:
            log.warning("Fichier historique corrompu : %s", e)
    return {}


def _save_local_history_legacy(history: dict[str, dict]):
    tmp = HISTORICAL_DATA_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, ensure_ascii=False, default=str)
    tmp.replace(HISTORICAL_DATA_FILE)


# ── Aliases pour compatibilite avec app.py (auto_feed) ──
def _load_local_history() -> dict[str, dict]:
    """Alias compatible — charge depuis Supabase via cache."""
    return _load_history()


def _save_local_history(history: dict[str, dict]):
    """
    Alias compatible — sauvegarde via Supabase.
    Calcule le delta vs le cache disque existant et appelle `append_matches`
    pour ECRIRE reellement les nouvelles lignes sur Supabase
    (et pas juste mettre a jour le cache local).
    """
    try:
        from supabase_db import _load_local_cache, append_matches, _save_local_cache
        existing = _load_local_cache()
        new_matches = {fid: data for fid, data in history.items() if fid not in existing}
        if new_matches:
            # Ecrit sur Supabase + met a jour le cache local
            append_matches(new_matches, existing)
        else:
            # Rien de neuf : on rafraichit juste le cache local
            _save_local_cache(history)
        return
    except ImportError:
        pass
    except Exception as e:
        log.warning("Erreur sync Supabase (append_matches) : %s — fallback JSON", e)

    # Fallback : ancien systeme JSON
    _save_local_history_legacy(history)


# ══════════════════════════════════════════════
# 1. CONSTRUCTION DU DATASET D'ENTRAINEMENT
# ══════════════════════════════════════════════
def _label_fn(target: str):
    """Retourne une fonction (hg, ag) -> label pour une cible donnée."""
    if target == "over25":
        return lambda hg, ag: 1 if (hg + ag) > 2 else 0
    if target == "over15":
        return lambda hg, ag: 1 if (hg + ag) > 1 else 0
    if target == "btts":
        return lambda hg, ag: 1 if (hg > 0 and ag > 0) else 0
    if target == "winner":
        # 0 = Home win, 1 = Draw, 2 = Away win
        return lambda hg, ag: 0 if hg > ag else (1 if hg == ag else 2)
    raise ValueError(f"Cible inconnue : {target}")


def build_training_dataset(historical_matches: list[dict],
                           target: str = "over25") -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit X (features) et y (label) à partir de matchs historiques.
    target ∈ {"over25", "over15", "btts", "winner"}
    """
    model_cols = get_model_columns()
    fn = _label_fn(target)
    rows = []
    labels = []

    for match in historical_matches:
        try:
            vec = build_feature_vector(match)
            if vec is None:
                continue

            result = match.get("result", {})
            hg = result.get("home_goals")
            ag = result.get("away_goals")
            if hg is None or ag is None:
                continue

            label = fn(int(hg), int(ag))
            row = {col: vec.get(col, 0.0) for col in model_cols}
            rows.append(row)
            labels.append(label)
        except Exception as e:
            log.warning("Erreur feature extraction pour match %s : %s",
                        match.get("fixture_id", "?"), e)
            continue

    X = pd.DataFrame(rows, columns=model_cols)
    y = pd.Series(labels, name=target)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if target == "winner":
        rate_str = f"H={(y==0).mean()*100:.1f}% D={(y==1).mean()*100:.1f}% A={(y==2).mean()*100:.1f}%"
    else:
        rate_str = f"{target}_rate = {y.mean()*100:.2f}%"
    log.info("Dataset [%s] : %d matchs, %d features, %s",
             target, len(X), len(model_cols), rate_str)

    return X, y


# ══════════════════════════════════════════════
# 2. MODELES INDIVIDUELS
# ══════════════════════════════════════════════
def _build_xgb() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _build_lgb() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=1.2,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ══════════════════════════════════════════════
# 3. ENSEMBLE CALIBRE
# ══════════════════════════════════════════════
class SimbetEnsemble:
    """
    Ensemble XGBoost + LightGBM.
    - Moyenne ponderee des probabilites
    - Calibration isotonique pour des probas fiables
    - Scaler integre
    """

    def __init__(self, xgb_weight: float = 0.5, lgb_weight: float = 0.5):
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.scaler = StandardScaler()
        self.is_calibrated = False
        self.calibrated_xgb = None
        self.calibrated_lgb = None
        self.feature_columns = get_model_columns()
        self.feature_importances_ = {}
        self.metrics_ = {}
        self.trained_date = None

    def train(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True):
        """
        Entraine l'ensemble sur le dataset complet.
        Utilise TimeSeriesSplit pour la validation.
        """
        log.info("Entrainement Smart Sim Ensemble sur %d matchs…", len(X))

        X = X[self.feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]

        log.info("Train: %d, Eval: %d", len(X_train), len(X_eval))

        # ── XGBoost ──
        self.xgb_model = _build_xgb()
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        # ── LightGBM ──
        self.lgb_model = _build_lgb()
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # ── Calibration ──
        if calibrate and len(X_eval) >= 50:
            log.info("Calibration isotonique…")
            self.calibrated_xgb = _make_calibrated(self.xgb_model, "isotonic")
            self.calibrated_xgb.fit(X_eval, y_eval)

            self.calibrated_lgb = _make_calibrated(self.lgb_model, "isotonic")
            self.calibrated_lgb.fit(X_eval, y_eval)
            self.is_calibrated = True
        else:
            self.is_calibrated = False

        # ── Feature Importances ──
        xgb_imp = dict(zip(self.feature_columns, self.xgb_model.feature_importances_))
        lgb_imp = dict(zip(self.feature_columns, self.lgb_model.feature_importances_))

        xgb_total = sum(xgb_imp.values()) or 1
        lgb_total = sum(lgb_imp.values()) or 1
        self.feature_importances_ = {
            col: round(
                (xgb_imp.get(col, 0) / xgb_total * self.xgb_weight +
                 lgb_imp.get(col, 0) / lgb_total * self.lgb_weight),
                6,
            )
            for col in self.feature_columns
        }

        # ── Metriques sur eval ──
        y_proba_eval = self._predict_proba_internal(X_eval)
        y_pred_eval = (y_proba_eval >= 0.5).astype(int)

        self.metrics_ = {
            "accuracy": round(accuracy_score(y_eval, y_pred_eval), 4),
            "precision": round(precision_score(y_eval, y_pred_eval, zero_division=0), 4),
            "recall": round(recall_score(y_eval, y_pred_eval, zero_division=0), 4),
            "f1": round(f1_score(y_eval, y_pred_eval, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_eval, y_proba_eval), 4),
            "brier_score": round(brier_score_loss(y_eval, y_proba_eval), 4),
            "log_loss": round(log_loss(y_eval, y_proba_eval), 4),
            "eval_size": len(y_eval),
            "train_size": len(y_train),
            "over25_rate_train": round(y_train.mean(), 4),
            "over25_rate_eval": round(y_eval.mean(), 4),
        }

        self.trained_date = datetime.now().isoformat()

        log.info("=== Metriques Eval ===")
        for k, v in self.metrics_.items():
            log.info("  %s : %s", k, v)

        # ── Cross-validation TimeSeriesSplit ──
        try:
            tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 50)))
            xgb_cv = cross_val_score(
                _build_xgb_no_early(), X_scaled, y, cv=tscv, scoring="roc_auc"
            )
            lgb_cv = cross_val_score(
                _build_lgb_no_early(), X_scaled, y, cv=tscv, scoring="roc_auc"
            )
            self.metrics_["cv_xgb_auc_mean"] = round(xgb_cv.mean(), 4)
            self.metrics_["cv_xgb_auc_std"] = round(xgb_cv.std(), 4)
            self.metrics_["cv_lgb_auc_mean"] = round(lgb_cv.mean(), 4)
            self.metrics_["cv_lgb_auc_std"] = round(lgb_cv.std(), 4)
            log.info("CV XGB AUC: %.4f +/- %.4f", xgb_cv.mean(), xgb_cv.std())
            log.info("CV LGB AUC: %.4f +/- %.4f", lgb_cv.mean(), lgb_cv.std())
        except Exception as e:
            log.warning("Cross-validation echouee (dataset trop petit?) : %s", e)

    def _predict_proba_internal(self, X_scaled: pd.DataFrame) -> np.ndarray:
        if self.is_calibrated:
            p_xgb = self.calibrated_xgb.predict_proba(X_scaled)[:, 1]
            p_lgb = self.calibrated_lgb.predict_proba(X_scaled)[:, 1]
        else:
            p_xgb = self.xgb_model.predict_proba(X_scaled)[:, 1]
            p_lgb = self.lgb_model.predict_proba(X_scaled)[:, 1]

        return self.xgb_weight * p_xgb + self.lgb_weight * p_lgb

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_columns].copy()
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return self._predict_proba_internal(X_scaled)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_match(self, features: dict) -> dict:
        row = {col: features.get(col, 0.0) for col in self.feature_columns}
        X = pd.DataFrame([row], columns=self.feature_columns)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        proba = float(self.predict_proba(X)[0])
        pred = 1 if proba >= MIN_CONFIDENCE else 0

        smart = detect_smart_bet(proba, features)

        from features import get_top_drivers
        drivers = get_top_drivers(features, self.feature_importances_, top_n=5)

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        if self.is_calibrated:
            p_xgb = float(self.calibrated_xgb.predict_proba(X_scaled)[0, 1])
            p_lgb = float(self.calibrated_lgb.predict_proba(X_scaled)[0, 1])
        else:
            p_xgb = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
            p_lgb = float(self.lgb_model.predict_proba(X_scaled)[0, 1])

        return {
            "proba_over25": round(proba, 4),
            "prediction": pred,
            "label": "OVER 2.5" if pred == 1 else "UNDER 2.5",
            "confidence": round(proba if pred == 1 else 1 - proba, 4),
            "xgb_proba": round(p_xgb, 4),
            "lgb_proba": round(p_lgb, 4),
            "smart_bet": smart,
            "top_drivers": drivers,
        }

    def save(self, tag: str = "latest"):
        """Sauvegarde le modele complet sur disque (pickle + joblib + JSON)."""
        # ── Pickle (format principal) ──
        path_pkl = MODELS_DIR / f"simbet_ensemble_{tag}.pkl"
        state = {
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "scaler": self.scaler,
            "is_calibrated": self.is_calibrated,
            "calibrated_xgb": self.calibrated_xgb,
            "calibrated_lgb": self.calibrated_lgb,
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
            "feature_columns": self.feature_columns,
            "feature_importances": self.feature_importances_,
            "metrics": self.metrics_,
            "trained_date": self.trained_date,
        }
        with open(path_pkl, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ── Joblib (compatibilite) ──
        # Pour tag="latest" : conserve l'historique `model.joblib` (jumeau du _latest.pkl).
        # Pour tout autre tag (candidate, expérimentation) : suffixe `model_{tag}.joblib`
        # pour ne JAMAIS écraser le miroir du modèle production.
        if tag == "latest":
            path_joblib = MODELS_DIR / "model.joblib"
        else:
            path_joblib = MODELS_DIR / f"model_{tag}.joblib"
        with open(path_joblib, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ── Metriques JSON ──
        metrics_path = MODELS_DIR / f"metrics_{tag}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_, f, indent=2)

        # ── Verification ──
        if path_pkl.exists() and path_joblib.exists():
            log.info("Modele sauvegarde : %s + %s", path_pkl, path_joblib)
        else:
            log.error("Echec sauvegarde modele !")

        return path_pkl

    @classmethod
    def load(cls, tag: str = "latest") -> "SimbetEnsemble":
        path = MODELS_DIR / f"simbet_ensemble_{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Modele introuvable : {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        ensemble = cls(
            xgb_weight=state["xgb_weight"],
            lgb_weight=state["lgb_weight"],
        )
        ensemble.xgb_model = state["xgb_model"]
        ensemble.lgb_model = state["lgb_model"]
        ensemble.scaler = state["scaler"]
        ensemble.is_calibrated = state["is_calibrated"]
        ensemble.calibrated_xgb = state["calibrated_xgb"]
        ensemble.calibrated_lgb = state["calibrated_lgb"]
        ensemble.feature_columns = state["feature_columns"]
        ensemble.feature_importances_ = state["feature_importances"]
        ensemble.metrics_ = state["metrics"]
        ensemble.trained_date = state.get("trained_date")

        log.info("Modele charge : %s (entraine le %s)", path, ensemble.trained_date)
        return ensemble

    @classmethod
    def exists(cls, tag: str = "latest") -> bool:
        return (MODELS_DIR / f"simbet_ensemble_{tag}.pkl").exists()


# ══════════════════════════════════════════════
# 4. VERSIONS SANS EARLY STOPPING (pour CV)
# ══════════════════════════════════════════════
def _build_xgb_no_early() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _build_lgb_no_early() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=1.2,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ══════════════════════════════════════════════
# 5. TRAINING PIPELINE
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# WINNER ENSEMBLE (1X2 multiclasse)
# ══════════════════════════════════════════════
class WinnerEnsemble:
    """
    Ensemble XGBoost multiclasse pour la prédiction 1X2 (Home/Draw/Away).
    Plus léger que SimbetEnsemble — pas de calibration isotonique multiclasse
    fiable, donc on garde XGB seul avec scaling.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.feature_columns = get_model_columns()
        self.metrics_ = {}
        self.trained_date = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        log.info("Entrainement WinnerEnsemble (1X2) sur %d matchs…", len(X))
        X = X[self.feature_columns].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X),
                                 columns=X.columns, index=X.index)
        split_idx = int(len(X) * 0.8)
        X_train, X_eval = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", early_stopping_rounds=50,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

        # Métriques
        y_pred = self.model.predict(X_eval)
        self.metrics_ = {
            "accuracy": round(accuracy_score(y_eval, y_pred), 4),
            "eval_size": len(y_eval),
            "train_size": len(y_train),
            "home_rate_train": round((y_train == 0).mean(), 4),
            "draw_rate_train": round((y_train == 1).mean(), 4),
            "away_rate_train": round((y_train == 2).mean(), 4),
        }
        self.trained_date = datetime.now().isoformat()
        log.info("Winner Eval Accuracy: %.4f", self.metrics_["accuracy"])

    def predict_proba(self, features: dict) -> dict:
        """Retourne {home: p, draw: p, away: p}."""
        row = {col: features.get(col, 0.0) for col in self.feature_columns}
        X = pd.DataFrame([row], columns=self.feature_columns)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_scaled = pd.DataFrame(self.scaler.transform(X),
                                 columns=X.columns, index=X.index)
        proba = self.model.predict_proba(X_scaled)[0]
        return {
            "home": float(proba[0]),
            "draw": float(proba[1]),
            "away": float(proba[2]),
        }

    def save(self, tag: str = "latest"):
        path = MODELS_DIR / f"simbet_winner_{tag}.pkl"
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics_,
            "trained_date": self.trained_date,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Winner model sauvegardé : %s", path)
        return path

    @classmethod
    def load(cls, tag: str = "latest") -> "WinnerEnsemble":
        path = MODELS_DIR / f"simbet_winner_{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Winner model introuvable : {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        ens = cls()
        ens.model = state["model"]
        ens.scaler = state["scaler"]
        ens.feature_columns = state["feature_columns"]
        ens.metrics_ = state["metrics"]
        ens.trained_date = state.get("trained_date")
        return ens

    @classmethod
    def exists(cls, tag: str = "latest") -> bool:
        return (MODELS_DIR / f"simbet_winner_{tag}.pkl").exists()


# ══════════════════════════════════════════════
# ORCHESTRATEUR : entraîne TOUS les marchés
# ══════════════════════════════════════════════
def train_all_markets(historical_matches: list[dict], base_tag: str = "latest") -> dict:
    """
    Entraîne 4 modèles : Over 2.5, Over 1.5, BTTS (binaires) + Winner (multiclasse).
    Chaque modèle est sauvegardé séparément.
    Retourne un dict des modèles entraînés.
    """
    if not historical_matches:
        raise ValueError("Aucune donnée historique fournie.")

    models = {}

    # 1. Over 2.5 (modèle principal — historique compatible)
    log.info("=== TRAIN [over25] ===")
    X, y = build_training_dataset(historical_matches, target="over25")
    if len(X) >= 50:
        ens = SimbetEnsemble(xgb_weight=0.5, lgb_weight=0.5)
        ens.train(X, y, calibrate=len(X) >= 200)
        ens.save(f"{base_tag}")  # tag historique = "latest"
        models["over25"] = ens

    # 2. Over 1.5
    log.info("=== TRAIN [over15] ===")
    X15, y15 = build_training_dataset(historical_matches, target="over15")
    if len(X15) >= 50 and y15.nunique() > 1:
        ens15 = SimbetEnsemble(xgb_weight=0.5, lgb_weight=0.5)
        ens15.train(X15, y15, calibrate=len(X15) >= 200)
        ens15.save(f"over15_{base_tag}")
        models["over15"] = ens15

    # 3. BTTS
    log.info("=== TRAIN [btts] ===")
    Xb, yb = build_training_dataset(historical_matches, target="btts")
    if len(Xb) >= 50 and yb.nunique() > 1:
        ens_btts = SimbetEnsemble(xgb_weight=0.5, lgb_weight=0.5)
        ens_btts.train(Xb, yb, calibrate=len(Xb) >= 200)
        ens_btts.save(f"btts_{base_tag}")
        models["btts"] = ens_btts

    # 4. Winner (1X2)
    log.info("=== TRAIN [winner 1X2] ===")
    Xw, yw = build_training_dataset(historical_matches, target="winner")
    if len(Xw) >= 50 and yw.nunique() == 3:
        winner = WinnerEnsemble()
        winner.train(Xw, yw)
        winner.save(base_tag)
        models["winner"] = winner

    log.info("=== Train all markets : %d modèles entraînés ===", len(models))
    # Invalider le cache pour forcer le rechargement des nouveaux modèles
    invalidate_model_cache()
    return models


def train_from_history(historical_matches: list[dict], tag: str = "latest") -> SimbetEnsemble:
    """
    Pipeline complet : donnees historiques → modele entraine + sauvegarde.
    """
    X, y = build_training_dataset(historical_matches)

    if len(X) == 0:
        raise ValueError("Aucune donnee d'entrainement valide.")

    if len(X) < 100:
        log.warning("Dataset petit (%d matchs). Le modele sera moins precis.", len(X))

    ensemble = SimbetEnsemble(xgb_weight=0.5, lgb_weight=0.5)

    try:
        ensemble.train(X, y, calibrate=len(X) >= 200)
    except Exception as e:
        log.error("Erreur pendant l'entrainement : %s", e)
        log.info("Tentative d'entrainement simplifie (sans calibration)…")
        ensemble.train(X, y, calibrate=False)

    # Sauvegarde GARANTIE
    ensemble.save(tag)

    # Verification finale — chemins joblib alignés avec le tag (cf. SimbetEnsemble.save)
    joblib_path = MODELS_DIR / ("model.joblib" if tag == "latest" else f"model_{tag}.joblib")
    pkl_path = MODELS_DIR / f"simbet_ensemble_{tag}.pkl"
    if pkl_path.exists() and joblib_path.exists():
        log.info("Modele valide : pkl=%s, joblib=%s", pkl_path, joblib_path)
    else:
        log.error("ATTENTION : fichier modele manquant apres sauvegarde !")

    return ensemble


# ══════════════════════════════════════════════
# 6. FETCH HISTORIQUE AVEC REPRISE + SAUVEGARDE TEMPS REEL
# ══════════════════════════════════════════════
def fetch_historical_for_training(league_id: int, season: int,
                                   local_history: dict = None) -> tuple[list[dict], bool]:
    """
    Recupere les matchs termines d'une ligue pour construire le dataset.

    REGLES :
    1. Verifie le cache (Supabase via supabase_db) AVANT chaque appel API
    2. Pause de 1.2s entre chaque appel (anti-ban)
    3. Sauvegarde apres chaque match enrichi (Supabase + cache local)
    4. Si quota/rate-limit → arret propre, retourne ce qu'on a
    """
    from data_fetcher import (
        api_get, fetch_fixture_stats, fetch_fixture_events,
        extract_goal_timings, fetch_last_matches_with_stats,
        fetch_goal_timings_for_matches, fetch_h2h_with_stats,
        fetch_referee_stats, extract_referee_name, fetch_odds_over25,
        get_market_consensus, CACHE_TTL_STATS, QuotaExceeded,
    )

    if local_history is None:
        local_history = _load_history()

    # Recuperer la liste des matchs termines de la saison
    try:
        data = api_get("fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT",
        }, ttl=CACHE_TTL_STATS)
    except QuotaExceeded:
        log.warning("Quota atteint des le fetch fixtures. Arret.")
        return [], False

    if not data or not data.get("response"):
        return [], True

    all_fixtures = data["response"]
    log.info("Ligue %d saison %d : %d matchs termines.", league_id, season, len(all_fixtures))

    all_fixtures.sort(key=lambda x: x.get("fixture", {}).get("date", ""))

    skip = LAST_N_MATCHES
    usable = all_fixtures[skip:]

    historical = []
    new_matches_buffer = {}  # Buffer pour batch append sur Supabase

    for i, fixture in enumerate(usable):
        fixture_info = fixture.get("fixture", {})
        fixture_id = fixture_info.get("id")
        home_team = fixture.get("teams", {}).get("home", {})
        away_team = fixture.get("teams", {}).get("away", {})
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        home_goals = fixture.get("goals", {}).get("home")
        away_goals = fixture.get("goals", {}).get("away")

        if not all([fixture_id, home_id, away_id, home_goals is not None, away_goals is not None]):
            continue

        fid_str = str(fixture_id)

        # ── REGLE 1 : CACHE — deja traite ? ──
        if fid_str in local_history:
            historical.append(local_history[fid_str])
            continue

        # ── Ce match n'est pas en cache, on le fetch ──
        match_date = fixture_info.get("date", "")
        log.info("  [%d/%d] Fetch fixture %d : %s vs %s…",
                 i + 1, len(usable), fixture_id,
                 home_team.get("name", "?"), away_team.get("name", "?"))

        try:
            home_last = fetch_last_matches_before(home_id, match_date, LAST_N_MATCHES)
            away_last = fetch_last_matches_before(away_id, match_date, LAST_N_MATCHES)

            home_last = _enrich_matches(home_last)
            away_last = _enrich_matches(away_last)

            h2h = fetch_h2h_with_stats(home_id, away_id)
            h2h = fetch_goal_timings_for_matches(h2h)

            referee_name = extract_referee_name(fixture)
            referee_data = fetch_referee_stats(referee_name, season) if referee_name else None

            odds = fetch_odds_over25(fixture_id)
            market = get_market_consensus(odds)

            home_rest = _calc_rest_from_history(home_last, match_date)
            away_rest = _calc_rest_from_history(away_last, match_date)

        except QuotaExceeded:
            log.warning("Quota/Rate-limit atteint au match %d/%d. Arret propre.",
                        i + 1, len(usable))
            # Sauvegarder ce qu'on a avant de quitter
            if new_matches_buffer:
                _save_history(local_history, new_matches_buffer)
            return historical, False

        except Exception as e:
            log.warning("Erreur sur fixture %d, on skip : %s", fixture_id, e)
            time.sleep(0.5)
            continue

        match_data = {
            "fixture_id": fixture_id,
            "league_id": league_id,
            "league_name": LEAGUES.get(league_id, {}).get("name", ""),
            "date": match_date,
            "home_team": {"id": home_id, "name": home_team.get("name"), "logo": home_team.get("logo")},
            "away_team": {"id": away_id, "name": away_team.get("name"), "logo": away_team.get("logo")},
            "home_last_matches": home_last,
            "away_last_matches": away_last,
            "h2h": h2h,
            "referee": referee_data,
            "odds": market,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "result": {
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            },
        }
        historical.append(match_data)

        # ── REGLE 3 : SAUVEGARDE TEMPS REEL ──
        local_history[fid_str] = match_data
        new_matches_buffer[fid_str] = match_data

        # Flush buffer tous les 10 matchs
        if len(new_matches_buffer) >= 10:
            _save_history(local_history, new_matches_buffer)
            new_matches_buffer = {}

    # Flush final
    if new_matches_buffer:
        _save_history(local_history, new_matches_buffer)

    log.info("Dataset ligue %d : %d matchs exploitables.", league_id, len(historical))
    return historical, True


def fetch_last_matches_before(team_id: int, before_date: str, n: int) -> list[dict]:
    """
    Recupere les derniers matchs d'une equipe AVANT une date donnee.
    """
    from data_fetcher import api_get, CACHE_TTL_STATS

    data = api_get("fixtures", {
        "team": team_id,
        "last": n + 5,
    }, ttl=CACHE_TTL_STATS)

    if not data:
        return []

    matches = data.get("response", [])
    filtered = [
        m for m in matches
        if m.get("fixture", {}).get("date", "") < before_date
    ]
    return filtered[:n]


def _enrich_matches(matches: list[dict]) -> list[dict]:
    from data_fetcher import fetch_fixture_stats, fetch_fixture_events, extract_goal_timings

    enriched = []
    for match in matches:
        fixture_id = match.get("fixture", {}).get("id")
        m = dict(match)
        if fixture_id:
            try:
                m["match_stats"] = fetch_fixture_stats(fixture_id)
                events = fetch_fixture_events(fixture_id)
                m["goal_timings"] = extract_goal_timings(events)
            except Exception as e:
                log.warning("Erreur enrichissement fixture %s : %s", fixture_id, e)
                m["match_stats"] = None
                m["goal_timings"] = {"first_goal_minute": None, "goals_last_15": 0,
                                      "goal_minutes": [], "total_goals": 0}
        enriched.append(m)
    return enriched


def _calc_rest_from_history(matches: list[dict], ref_date: str) -> Optional[int]:
    if not matches:
        return None
    try:
        ref_dt = datetime.fromisoformat(ref_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    for match in matches:
        md = match.get("fixture", {}).get("date", "")
        if md:
            try:
                m_dt = datetime.fromisoformat(md.replace("Z", "+00:00"))
                delta = (ref_dt - m_dt).days
                if delta >= 0:
                    return delta
            except (ValueError, AttributeError):
                continue
    return None


# ══════════════════════════════════════════════
# 7. QUICK TRAIN (source : Supabase)
# ══════════════════════════════════════════════
def quick_train(league_ids: Optional[list[int]] = None, tag: str = "latest") -> SimbetEnsemble:
    """
    Entrainement avec reprise automatique.
    1. Charge l'historique depuis Supabase (via supabase_db + cache)
    2. Fetch uniquement les matchs manquants
    3. Si quota atteint → entraine avec ce qu'on a
    4. Sauvegarde pkl + joblib + json
    """
    if league_ids is None:
        league_ids = list(LEAGUES.keys())

    # ── Charger l'historique (Supabase + cache) ──
    local_history = _load_history()
    already_cached = len(local_history)
    log.info("=== Smart Sim — Entrainement ===")
    log.info("Historique charge : %d matchs (Supabase + cache).", already_cached)

    all_historical = []
    quota_ok = True

    for lid in league_ids:
        meta = LEAGUES.get(lid, {})
        season = meta.get("season", 2025)
        log.info("-- Ligue : %s %s (saison %d) --", meta.get("flag", ""), meta.get("name", lid), season)

        matches, still_ok = fetch_historical_for_training(lid, season, local_history)
        all_historical.extend(matches)
        log.info("  -> %d matchs (total : %d)", len(matches), len(all_historical))

        if not still_ok:
            quota_ok = False
            log.warning("Quota atteint. Arret du fetch, on entraine avec %d matchs.", len(all_historical))
            break

    # ── Sauvegarde finale ──
    new_cached = len(local_history) - already_cached
    log.info("Historique mis a jour : +%d nouveaux matchs (total : %d)", new_cached, len(local_history))

    # ── Entrainement (meme si partiel) ──
    if not all_historical:
        raise ValueError("Aucune donnee historique recuperee (cache vide + API inaccessible).")

    if not quota_ok:
        log.info("Entrainement sur donnees partielles (%d matchs)…", len(all_historical))

    # Entraîne TOUS les marchés (Over 2.5, Over 1.5, BTTS, Winner 1X2)
    models = train_all_markets(all_historical, base_tag=tag)
    # Compatibilité : retourner le modèle Over 2.5 (le principal)
    return models.get("over25") or train_from_history(all_historical, tag)


# ══════════════════════════════════════════════
# 8. PREDICTION BATCH (pour l'UI)
# ══════════════════════════════════════════════
# Cache module-level des modèles ML : chargés UNE SEULE FOIS par process Python
# Évite les rechargements coûteux à chaque appel de predict_today
_MODEL_CACHE = {}


def _get_cached_models(tag: str = "latest"):
    """
    Retourne (over25, over15, btts, winner) avec cache module-level.
    Les modèles ne sont chargés qu'une fois par process Python.
    """
    if tag in _MODEL_CACHE:
        return _MODEL_CACHE[tag]

    if not SimbetEnsemble.exists(tag):
        return None, None, None, None

    over25 = SimbetEnsemble.load(tag)
    over15 = None
    btts = None
    winner = None

    try:
        if SimbetEnsemble.exists(f"over15_{tag}"):
            over15 = SimbetEnsemble.load(f"over15_{tag}")
            log.info("Modèle Over 1.5 chargé (cache)")
    except Exception as e:
        log.warning("Modèle Over 1.5 non chargé : %s", e)
    try:
        if SimbetEnsemble.exists(f"btts_{tag}"):
            btts = SimbetEnsemble.load(f"btts_{tag}")
            log.info("Modèle BTTS chargé (cache)")
    except Exception as e:
        log.warning("Modèle BTTS non chargé : %s", e)
    try:
        if WinnerEnsemble.exists(tag):
            winner = WinnerEnsemble.load(tag)
            log.info("Modèle Winner 1X2 chargé (cache)")
    except Exception as e:
        log.warning("Modèle Winner non chargé : %s", e)

    _MODEL_CACHE[tag] = (over25, over15, btts, winner)
    return _MODEL_CACHE[tag]


def invalidate_model_cache():
    """Vide le cache modèles (à appeler après réentrainement)."""
    _MODEL_CACHE.clear()
    log.info("Cache modèles vidé")


def predict_today(matches_data: list[dict], tag: str = "latest") -> list[dict]:
    ensemble, over15_model, btts_model, winner_model = _get_cached_models(tag)
    if ensemble is None:
        log.error("Aucun modele trouve. Lancer quick_train() d'abord.")
        return []

    results = []

    for match in matches_data:
        try:
            vec = build_feature_vector(match)
            if vec is None:
                continue

            prediction = ensemble.predict_match(vec)
            # Prédictions ML auxiliaires (None si modèle absent)
            if over15_model is not None:
                vec["ml_proba_o15"] = float(over15_model.predict_proba(
                    pd.DataFrame([{c: vec.get(c, 0.0) for c in over15_model.feature_columns}])
                )[0])
            if btts_model is not None:
                vec["ml_proba_btts"] = float(btts_model.predict_proba(
                    pd.DataFrame([{c: vec.get(c, 0.0) for c in btts_model.feature_columns}])
                )[0])
            if winner_model is not None:
                wp = winner_model.predict_proba(vec)
                vec["ml_winner_home"] = wp["home"]
                vec["ml_winner_draw"] = wp["draw"]
                vec["ml_winner_away"] = wp["away"]

            results.append({
                "fixture_id": match["fixture_id"],
                "league_id": match["league_id"],
                "league_name": match.get("league_name", ""),
                "league_flag": match.get("league_flag", ""),
                "league_country": match.get("league_country", ""),
                "date": match.get("date", ""),
                "venue": match.get("venue", ""),
                # ── Statut & score live/final ──
                "match_status": match.get("match_status", "NS"),
                "match_elapsed": match.get("match_elapsed"),
                "current_home_goals": match.get("current_home_goals"),
                "current_away_goals": match.get("current_away_goals"),
                # ── Equipes ──
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "prediction": prediction,
                "features": vec,
                "odds_data": match.get("odds"),
                "odd_over15": match.get("odd_over15"),
                "odd_btts": match.get("odd_btts"),
                "odds_fetched_at": match.get("odds_fetched_at"),
                "referee_data": match.get("referee"),
                "lineups": match.get("lineups"),
                "injuries": match.get("injuries", []),
                "home_missing_players": match.get("home_missing_players", 0),
                "away_missing_players": match.get("away_missing_players", 0),
                # ── Historique forme (conservé pour archivage ML) ──
                "home_last_matches": match.get("home_last_matches", []),
                "away_last_matches": match.get("away_last_matches", []),
                "h2h": match.get("h2h", []),
            })
        except Exception as e:
            log.warning("Erreur prediction match %s : %s", match.get("fixture_id"), e)
            continue

    results.sort(key=lambda x: x["prediction"]["proba_over25"], reverse=True)

    log.info("Predictions : %d matchs, %d OVER, %d SMART BET",
             len(results),
             sum(1 for r in results if r["prediction"]["prediction"] == 1),
             sum(1 for r in results if r["prediction"]["smart_bet"]["is_smart_bet"]))

    return results
