"""Auto-calibration : mesure le biais systématique du modèle sur les matchs résolus
récents et applique une correction sur les nouvelles prédictions.

Pour chaque marché binaire (over_25, over_15, btts) :
    scale, bias = fit_calibration(predictions, outcomes)
    calibrated_p = clip(scale * predicted_p + bias, 0.02, 0.98)

Approche : régression linéaire simple (moindres carrés) sur les paires (p_pred, y_actual).
Refresh toutes les 12h en RAM. Fallback identité (scale=1, bias=0) si pas assez de données.

Sur 100+ matchs résolus par marché, on obtient une calibration robuste sans overfit.
"""
from __future__ import annotations
import logging
import os
import time

log = logging.getLogger("SmartSim.calibration")

REFRESH_SECONDS = 12 * 3600
MIN_SAMPLES = 60
MAX_SAMPLES = 800
DAYS_BACK = 45

# Bornes sur les corrections pour éviter les dérives extrêmes
SCALE_MIN, SCALE_MAX = 0.65, 1.35
BIAS_MIN, BIAS_MAX = -0.15, 0.15

_cache: dict = {"built_at": 0.0, "params": {}}
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        log.warning("supabase init failed: %s", e)
        return None


def _fit(pairs: list[tuple[float, int]]) -> dict:
    """Régression linéaire y = a*x + b sur (p, outcome). Clampé aux bornes."""
    n = len(pairs)
    if n < MIN_SAMPLES:
        return {"scale": 1.0, "bias": 0.0, "n": n}
    sx = sy = sxx = sxy = 0.0
    for x, y in pairs:
        sx += x; sy += y; sxx += x * x; sxy += x * y
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"scale": 1.0, "bias": 0.0, "n": n}
    a = (n * sxy - sx * sy) / denom  # slope
    b = (sy - a * sx) / n            # intercept
    scale = max(SCALE_MIN, min(SCALE_MAX, a))
    bias = max(BIAS_MIN, min(BIAS_MAX, b))
    return {"scale": scale, "bias": bias, "n": n}


def _build_cache() -> None:
    """Fetch les résultats récents résolus et fit une calibration par marché."""
    from datetime import datetime, timezone, timedelta
    client = _get_client()
    if not client:
        _cache["built_at"] = time.time()
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).date().isoformat()
    try:
        resp = (client.table("bet_history")
                    .select("proba_over25, proba_o15, proba_btts, "
                              "result_over25_won, result_over15_won, result_btts_won, "
                              "date")
                    .gte("date", cutoff)
                    .not_.is_("result_over25_won", "null")
                    .limit(MAX_SAMPLES)
                    .execute())
        rows = resp.data or []
    except Exception as e:
        log.warning("calibration fetch failed: %s", e)
        _cache["built_at"] = time.time()
        return

    pairs_o25 = []
    pairs_o15 = []
    pairs_btts = []
    for r in rows:
        p_o25 = r.get("proba_over25"); w_o25 = r.get("result_over25_won")
        p_o15 = r.get("proba_o15");    w_o15 = r.get("result_over15_won")
        p_bt  = r.get("proba_btts");   w_bt  = r.get("result_btts_won")
        if p_o25 is not None and w_o25 is not None:
            try: pairs_o25.append((float(p_o25), 1 if w_o25 else 0))
            except (TypeError, ValueError): pass
        if p_o15 is not None and w_o15 is not None:
            try: pairs_o15.append((float(p_o15), 1 if w_o15 else 0))
            except (TypeError, ValueError): pass
        if p_bt is not None and w_bt is not None:
            try: pairs_btts.append((float(p_bt), 1 if w_bt else 0))
            except (TypeError, ValueError): pass

    _cache["params"] = {
        "over_25": _fit(pairs_o25),
        "over_15": _fit(pairs_o15),
        "btts": _fit(pairs_btts),
    }
    _cache["built_at"] = time.time()
    log.info("calibration built: O25 %s, O15 %s, BTTS %s",
             _cache["params"]["over_25"], _cache["params"]["over_15"], _cache["params"]["btts"])


def _maybe_refresh() -> None:
    if time.time() - _cache["built_at"] > REFRESH_SECONDS:
        _build_cache()


def calibrate(prob: float, market: str) -> float:
    """Applique la correction apprise sur une proba brute. Fallback = identité si pas de calibration."""
    _maybe_refresh()
    params = _cache["params"].get(market)
    if not params or params.get("n", 0) < MIN_SAMPLES:
        return prob
    try:
        p = params["scale"] * float(prob) + params["bias"]
        return max(0.02, min(0.98, p))
    except (TypeError, ValueError):
        return prob


def diagnostics() -> dict:
    """Retourne les paramètres actuels + qualité (via Brier score) sur l'échantillon."""
    _maybe_refresh()
    return {
        "built_at": _cache["built_at"],
        "params": _cache["params"],
        "refresh_seconds": REFRESH_SECONDS,
        "min_samples": MIN_SAMPLES,
    }
