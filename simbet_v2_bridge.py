"""Lot 17A — Bridge SimBet_V2 (Lot 14 + 15 + 16) → API legacy SimBet V1.

Drop-in replacement for `model.predict_today(matches_data)`. Returns the
SAME list-of-dicts shape the old UI expects, but the `prediction` sub-dict
is populated from the V2 enricher (Lot 14 markets + Lot 15 safety +
Lot 16 combined payload).

Design constraints (immutable):
- No re-training, no re-calibration
- V2 frozen artefacts untouched
- V1 legacy modules (model.py / features.py) still importable
- Enrichment ledger append-only, separate from V2 immutable ledger

Usage from `app.py`:

    from simbet_v2_bridge import predict_today_v2 as predict_today
    results = predict_today(matches)  # same shape as legacy

Anti-freeze note:
The heavy artefacts (Lot 14 models + rolling state) are loaded ONCE
per process and cached module-level. Each fixture inference is <5 ms.
"""
from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("SimBet.bridge_v2")

_V2_ROOT = Path("/Users/simonfloret/Desktop/SIMBET_V2")
_V2_DATA = Path("/Users/simonfloret/Desktop/SIMBET_V2_DATA")

if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))


@functools.lru_cache(maxsize=1)
def _load_v2():
    from experiments.fast_track.lot16.enricher import (
        enrich_fixture, append_enrichments, STYLE_FEATS, ENRICH_PATH)
    live_style_p = _V2_DATA / "experiments/fast_track/lot13b5/live_style_dataset.parquet"
    style_df = pd.read_parquet(live_style_p) if live_style_p.exists() else pd.DataFrame()
    if len(style_df) > 0:
        style_df["fixture_id"] = style_df["fixture_id"].astype(str)
    base_p = _V2_DATA / "experiments/fast_track/dataset_player_large_scale.parquet"
    base_df = pd.read_parquet(base_p, columns=[
        "fixture_id","kickoff_utc","home_team_id","away_team_id",
        "home_elo_pre","away_elo_pre"])
    base_df["fixture_id"] = base_df["fixture_id"].astype(str)
    base_df["kickoff_utc"] = pd.to_datetime(base_df["kickoff_utc"], utc=True)
    base_df = base_df.sort_values("kickoff_utc")
    return {
        "enrich_fixture": enrich_fixture,
        "append_enrichments": append_enrichments,
        "STYLE_FEATS": STYLE_FEATS,
        "style_df": style_df,
        "base_df": base_df,
        "enrich_path": ENRICH_PATH,
    }


def _last_elo(base_df: pd.DataFrame, team_id: int, before: pd.Timestamp | None) -> float | None:
    if team_id is None: return None
    h = base_df[base_df["home_team_id"] == team_id]
    a = base_df[base_df["away_team_id"] == team_id]
    if before is not None:
        h = h[h["kickoff_utc"] < before]
        a = a[a["kickoff_utc"] < before]
    pool = pd.concat([
        h[["kickoff_utc","home_elo_pre"]].rename(columns={"home_elo_pre":"elo"}),
        a[["kickoff_utc","away_elo_pre"]].rename(columns={"away_elo_pre":"elo"})])
    if pool.empty: return None
    return float(pool.sort_values("kickoff_utc").iloc[-1]["elo"])


def _feature_vector(match: dict, ctx: dict) -> tuple[np.ndarray, float] | tuple[None, None]:
    STYLE_FEATS = ctx["STYLE_FEATS"]
    style_df = ctx["style_df"]
    base_df = ctx["base_df"]
    fx_id = str(match.get("fixture_id"))
    home_id = (match.get("home_team") or {}).get("id")
    away_id = (match.get("away_team") or {}).get("id")
    kickoff = match.get("date")
    try:
        kt = pd.to_datetime(kickoff, utc=True) if kickoff else None
    except Exception:
        kt = None
    h_elo = _last_elo(base_df, home_id, kt)
    a_elo = _last_elo(base_df, away_id, kt)
    if h_elo is None or a_elo is None:
        return None, None
    elo_diff = h_elo - a_elo
    vec = np.zeros(len(STYLE_FEATS))
    vec[0] = elo_diff
    if len(style_df) > 0:
        row = style_df[style_df["fixture_id"] == fx_id]
        if len(row) > 0:
            r = row.iloc[0]
            for i, f in enumerate(STYLE_FEATS[1:], start=1):
                v = r.get(f)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vec[i] = float(v)
    return vec, elo_diff


def _prediction_dict_from_enrichment(e: dict) -> dict:
    """Reshape a Lot 16 enrichment payload into the legacy `prediction` dict."""
    home_win = float(e.get("home_win_prob", 0.0))
    draw = float(e.get("draw_prob", 0.0))
    away_win = float(e.get("away_win_prob", 0.0))
    over25 = float(e.get("over_2_5_prob", 0.0))
    pred_over25 = int(over25 >= 0.5)
    return {
        "proba_over25": round(over25, 4),
        "proba_over15": round(float(e.get("over_1_5_prob", 0.0)), 4),
        "proba_over35": round(float(e.get("over_3_5_prob", 0.0)), 4),
        "proba_btts": round(float(e.get("btts_prob", 0.0)), 4),
        "proba_home": round(float(e.get("1x2_P_home", 0.0)), 4),
        "proba_draw": round(float(e.get("1x2_P_draw", 0.0)), 4),
        "proba_away": round(float(e.get("1x2_P_away", 0.0)), 4),
        "winner_prediction": int(e.get("1x2_prediction", 0)),
        "winner_label": ["HOME","DRAW","AWAY"][int(e.get("1x2_prediction", 0))],
        "exact_score_top1": e.get("exact_score_top1", ""),
        "exact_score_top3": e.get("exact_score_top3", ""),
        "exact_score_top5": e.get("exact_score_top5", ""),
        "exact_score_top1_prob": round(float(e.get("exact_score_top1_prob", 0.0)), 4),
        "prediction": pred_over25,
        "label": "OVER 2.5" if pred_over25 == 1 else "UNDER 2.5",
        "confidence": round(over25 if pred_over25 == 1 else 1 - over25, 4),
        "safety_score": float(e.get("safety_score", 0.0)),
        "safety_tier": e.get("safety_tier", "C"),
        "safety_badges": e.get("safety_badges", ""),
        "smart_bet": {
            "is_smart_bet": e.get("safety_tier") in ("S+", "S"),
            "reason": e.get("safety_badges", ""),
        },
        "top_drivers": [],
        "xgb_proba": round(over25, 4),
        "lgb_proba": round(over25, 4),
        "engine": "v2_lot16",
        "payload_hash": e.get("payload_hash", ""),
    }


def _implied(odd):
    """Convert decimal odd to implied probability. Returns 0 for invalid odds."""
    try:
        o = float(odd)
        return 1.0 / o if o > 1.01 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _normalize(triplet):
    """Normalize 3-way probas to sum to 1 (removes bookmaker margin)."""
    s = sum(triplet)
    if s <= 0: return (0.34, 0.33, 0.33)
    return tuple(x / s for x in triplet)


def _priors_from_form(match: dict) -> dict:
    """Estime des priors à partir des derniers matchs et H2H quand les cotes manquent.
    Pondère les matchs récents plus fortement (weight décroissant)."""
    def _team_weighted_avg(matches, team_id, key):
        # Poids décroissants : match récent = 1.0, ancien = 0.5
        weights_sum = 0.0
        goals_weighted = 0.0
        for i, past in enumerate((matches or [])[:10]):
            teams = past.get("teams") or {}
            home = teams.get("home") or {}; away = teams.get("away") or {}
            g = past.get("goals") or {}
            hg, ag = g.get("home"), g.get("away")
            if hg is None or ag is None: continue
            w = 1.0 - (i * 0.05)  # 1.0, 0.95, 0.90, ...
            if home.get("id") == team_id:
                v = hg if key == "scored" else ag
            elif away.get("id") == team_id:
                v = ag if key == "scored" else hg
            else:
                continue
            goals_weighted += v * w
            weights_sum += w
        return (goals_weighted / weights_sum) if weights_sum > 0 else None

    def _current_win_streak(matches, team_id):
        """Nombre de victoires consécutives en cours (0 si dernière ≠ V)."""
        streak = 0
        for past in (matches or []):
            teams = past.get("teams") or {}
            home = teams.get("home") or {}; away = teams.get("away") or {}
            g = past.get("goals") or {}
            hg, ag = g.get("home"), g.get("away")
            if hg is None or ag is None: continue
            if home.get("id") == team_id:
                won = hg > ag
            elif away.get("id") == team_id:
                won = ag > hg
            else:
                continue
            if won: streak += 1
            else: break
        return streak

    def _max_win_streak(matches, team_id):
        """Plus longue série de victoires dans les matchs disponibles."""
        best = cur = 0
        for past in (matches or []):
            teams = past.get("teams") or {}
            home = teams.get("home") or {}; away = teams.get("away") or {}
            g = past.get("goals") or {}
            hg, ag = g.get("home"), g.get("away")
            if hg is None or ag is None: continue
            if home.get("id") == team_id:
                won = hg > ag
            elif away.get("id") == team_id:
                won = ag > hg
            else:
                continue
            if won: cur += 1; best = max(best, cur)
            else: cur = 0
        return best

    home_id = (match.get("home_team") or {}).get("id")
    away_id = (match.get("away_team") or {}).get("id")
    home_last = match.get("home_last_matches") or match.get("home_last") or []
    away_last = match.get("away_last_matches") or match.get("away_last") or []
    home_scored = _team_weighted_avg(home_last, home_id, "scored")
    home_conc = _team_weighted_avg(home_last, home_id, "conceded")
    away_scored = _team_weighted_avg(away_last, away_id, "scored")
    away_conc = _team_weighted_avg(away_last, away_id, "conceded")

    # Combine attaque × défense adverse (formule Dixon-Coles simplifiée)
    lg_avg = 1.35
    home_attack = (home_scored or lg_avg) / lg_avg
    home_defense = (home_conc or lg_avg) / lg_avg
    away_attack = (away_scored or lg_avg) / lg_avg
    away_defense = (away_conc or lg_avg) / lg_avg

    # ── Régression vers la moyenne : uniquement pour séries exceptionnelles ──
    # Stats réelles : 79% des équipes n'atteignent PAS 3V d'affilée sur 5 matchs.
    # Donc on ne dampe QUE les équipes qui viennent de faire ≥3V (signal rare).
    def _regression_factor(cur_streak, max_streak):
        if cur_streak >= max_streak and max_streak >= 3:
            return 0.95  # -5% : série au max récent, régression attendue
        return 1.0

    home_cur = _current_win_streak(home_last, home_id)
    home_max = _max_win_streak(home_last, home_id)
    away_cur = _current_win_streak(away_last, away_id)
    away_max = _max_win_streak(away_last, away_id)
    home_attack *= _regression_factor(home_cur, home_max)
    away_attack *= _regression_factor(away_cur, away_max)

    lam_home = home_attack * away_defense * lg_avg * 1.10
    lam_away = away_attack * home_defense * lg_avg * 0.90
    lam_home = max(0.30, min(3.8, lam_home))
    lam_away = max(0.20, min(3.5, lam_away))
    # Approximation Poisson
    import math
    def _p_goals(l, k):
        return math.exp(-l) * (l ** k) / math.factorial(k)
    # P(home wins), draw, away wins via convolution jusqu'à 6-6
    p_h = p_d = p_a = 0.0
    p_o25 = p_o15 = p_btts = 0.0
    for i in range(7):
        for j in range(7):
            p = _p_goals(lam_home, i) * _p_goals(lam_away, j)
            if i > j: p_h += p
            elif i == j: p_d += p
            else: p_a += p
            if i + j >= 3: p_o25 += p
            if i + j >= 2: p_o15 += p
            if i >= 1 and j >= 1: p_btts += p
    return {"home": p_h, "draw": p_d, "away": p_a,
              "over_25": p_o25, "over_15": p_o15, "btts": p_btts,
              "source": "poisson-priors"}


def _predict_from_odds(match: dict) -> dict:
    """Fallback: derive prediction dict from bookmaker odds only.
    If odds are missing, fall back on Poisson estimation from recent form.
    """
    market = (match.get("odds") or match.get("market") or {})
    ext = market.get("extended_markets") or {}
    if not ext.get("over_15"):
        ext["over_15"] = match.get("odd_over15")
    if not ext.get("btts_yes"):
        ext["btts_yes"] = match.get("odd_btts")
    p_home, p_draw, p_away = _normalize((
        _implied(ext.get("home")), _implied(ext.get("draw")), _implied(ext.get("away"))))
    p_o25 = _implied(ext.get("over_25"))
    p_u25 = _implied(ext.get("under_25"))
    if p_o25 and p_u25:
        s = p_o25 + p_u25
        p_o25 /= s
    p_o15 = _implied(ext.get("over_15"))
    p_u15 = _implied(ext.get("under_15"))
    if p_o15 and p_u15:
        s = p_o15 + p_u15
        p_o15 /= s

    # Toujours calculer le modèle Poisson (forme) pour blend et détection value
    _model = _priors_from_form(match)
    has_odds_1x2 = p_home + p_draw + p_away > 0.98 and not (
        abs(p_home - 0.333) < 0.02 and abs(p_draw - 0.333) < 0.02)
    if has_odds_1x2:
        # Blend : 75% marché + 25% modèle → probas plus naturelles sans dévier trop
        p_home = 0.75 * p_home + 0.25 * _model["home"]
        p_draw = 0.75 * p_draw + 0.25 * _model["draw"]
        p_away = 0.75 * p_away + 0.25 * _model["away"]
        s = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    else:
        p_home, p_draw, p_away = _model["home"], _model["draw"], _model["away"]
    if p_o25 > 0:
        p_o25 = 0.75 * p_o25 + 0.25 * _model["over_25"]
    else:
        p_o25 = _model["over_25"]
    if p_o15 > 0:
        p_o15 = 0.75 * p_o15 + 0.25 * _model["over_15"]
    else:
        p_o15 = _model["over_15"]
    p_btts_bk = _implied(ext.get("btts_yes"))
    p_btts_no = _implied(ext.get("btts_no"))
    if p_btts_bk and p_btts_no:
        s = p_btts_bk + p_btts_no
        p_btts_bk /= s
        # Blend marché + modèle Poisson pour BTTS
        p_btts = 0.65 * p_btts_bk + 0.35 * _model["btts"]
    else:
        # Pas de cotes BTTS : moyenne Poisson + formule empirique
        p_btts_empirical = min(0.90, max(0.10, 0.55 * p_o25 + 0.25 * (1 - p_draw)))
        p_btts = 0.5 * _model["btts"] + 0.5 * p_btts_empirical
    p_btts = round(p_btts, 4)
    winner_idx = int(np.argmax([p_home, p_draw, p_away]))
    winner_conf = max(p_home, p_draw, p_away)

    # ── Détection de VALUE (edge modèle Poisson vs cote bookmaker) ──
    # Le modèle Poisson (_model) est indépendant des cotes → comparaison honnête.
    model_pick_idx = int(np.argmax([_model["home"], _model["draw"], _model["away"]]))
    model_pick_prob = [_model["home"], _model["draw"], _model["away"]][model_pick_idx]
    market_pick_implied = _implied(ext.get(["home", "draw", "away"][model_pick_idx]))
    winner_odd_raw = ext.get(["home", "draw", "away"][model_pick_idx])
    edge_winner = (model_pick_prob - market_pick_implied) if market_pick_implied > 0 else 0.0
    o25_implied = _implied(ext.get("over_25"))
    edge_o25 = (_model["over_25"] - o25_implied) if o25_implied > 0 else 0.0

    # Smart Sim = évidence solide OU value bet inattendu
    is_evidence = winner_conf >= 0.70 or (p_o25 >= 0.70 and winner_conf >= 0.50)
    # Value : edge > 8% avec proba raisonnable et cote attractive (1.7-4.5)
    try:
        odd_val = float(winner_odd_raw) if winner_odd_raw else 0
    except (TypeError, ValueError):
        odd_val = 0
    is_value_winner = (edge_winner >= 0.10 and model_pick_prob >= 0.42
                          and 1.70 <= odd_val <= 4.20)
    is_value_o25 = edge_o25 >= 0.10 and _model["over_25"] >= 0.55
    is_value = is_value_winner or is_value_o25
    is_smart = is_evidence or is_value

    if is_value and not is_evidence:
        reason = f"Value bet détectée : edge {round(max(edge_winner, edge_o25)*100)}% vs marché"
        tier = "S"
    elif is_evidence:
        reason = "Signal fort du modèle"
        tier = "S+"
    else:
        reason = "Prédiction dérivée des cotes du marché"
        tier = "A" if winner_conf >= 0.6 else "B"

    return {
        "proba_over25": round(p_o25, 4),
        "proba_over15": round(p_o15, 4),
        "proba_over35": 0.0,
        "proba_btts": round(p_btts, 4),
        "proba_home": round(p_home, 4),
        "proba_draw": round(p_draw, 4),
        "proba_away": round(p_away, 4),
        "winner_prediction": winner_idx,
        "winner_label": ["HOME","DRAW","AWAY"][winner_idx],
        "exact_score_top1": "", "exact_score_top3": "", "exact_score_top5": "",
        "exact_score_top1_prob": 0.0,
        "prediction": int(p_o25 >= 0.5),
        "label": "OVER 2.5" if p_o25 >= 0.5 else "UNDER 2.5",
        "confidence": round(p_o25 if p_o25 >= 0.5 else 1 - p_o25, 4),
        "safety_score": round(winner_conf, 3),
        "safety_tier": tier,
        "safety_badges": "value" if is_value and not is_evidence else "evidence" if is_evidence else "market",
        "edge": {
            "winner": round(edge_winner, 4),
            "over_25": round(edge_o25, 4),
        },
        "smart_bet": {
            "is_smart_bet": bool(is_smart),
            "is_value": bool(is_value and not is_evidence),
            "reason": reason,
        },
        "top_drivers": [], "xgb_proba": 0.0, "lgb_proba": 0.0,
        "engine": "odds_fallback",
        "payload_hash": "",
    }


def _extra_from_pred(pred: dict) -> dict:
    """Build the _extra dict expected by serialize_match_summary from the fallback pred."""
    p_home = pred.get("proba_home", 0.0)
    p_draw = pred.get("proba_draw", 0.0)
    p_away = pred.get("proba_away", 0.0)
    winner_p = max(p_home, p_draw, p_away)
    winner_label = ["Domicile", "Match nul", "Extérieur"][int(np.argmax([p_home, p_draw, p_away]))]
    return {
        "p_home_win": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
        "winner": winner_label,
        "winner_proba": round(winner_p, 4),
        "proba_o15": pred.get("proba_over15", 0.0),
        "proba_btts": pred.get("proba_btts", 0.0),
        "is_secure_bet": bool(pred.get("smart_bet", {}).get("is_smart_bet", False)),
    }


def _shape_result(match: dict, pred: dict) -> dict:
    return {
        "_extra": _extra_from_pred(pred),
        "fixture_id": match["fixture_id"],
        "league_id": match.get("league_id"),
        "league_name": match.get("league_name", ""),
        "league_flag": match.get("league_flag", ""),
        "league_country": match.get("league_country", ""),
        "date": match.get("date", ""),
        "venue": match.get("venue", ""),
        "match_status": match.get("match_status", "NS"),
        "match_elapsed": match.get("match_elapsed"),
        "current_home_goals": match.get("current_home_goals"),
        "current_away_goals": match.get("current_away_goals"),
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "prediction": pred,
        "features": {},
        "odds_data": match.get("odds"),
        "odd_over15": match.get("odd_over15"),
        "odd_btts": match.get("odd_btts"),
        "market": match.get("odds"),
        "referee": match.get("referee"),
        "lineups": match.get("lineups"),
        "injuries": match.get("injuries", []),
        # Correct keys used by fetch_full_match_data (was home_last, away_last)
        "home_last_matches": match.get("home_last_matches") or match.get("home_last") or [],
        "away_last_matches": match.get("away_last_matches") or match.get("away_last") or [],
        "home_euro_history": match.get("home_euro_history") or [],
        "away_euro_history": match.get("away_euro_history") or [],
        "h2h": match.get("h2h", []),
        "home_rest_days": match.get("home_rest_days"),
        "away_rest_days": match.get("away_rest_days"),
        "home_standing": match.get("home_standing"),
        "away_standing": match.get("away_standing"),
        "league_meta": match.get("league_meta"),
        "is_smart_bet": pred.get("smart_bet", {}).get("is_smart_bet", False),
    }


def predict_today_v2(matches_data: list[dict], tag: str = "latest",
                     persist: bool = True) -> list[dict]:
    """Drop-in replacement for `model.predict_today`.

    If V2 artefacts are available (local dev), use them. Otherwise fall back
    to a market-implied prediction derived from bookmaker odds.
    """
    try:
        ctx = _load_v2()
    except Exception as e:
        log.warning("[bridge_v2] V2 artefacts indispo (%s) — fallback cotes marché", e)
        return [_shape_result(m, _predict_from_odds(m)) for m in matches_data]
    results, to_persist = [], []
    for match in matches_data:
        try:
            vec, elo_diff = _feature_vector(match, ctx)
            if vec is None:
                continue
            league_id = int(match.get("league_id") or 0)
            fx_id = str(match.get("fixture_id"))
            # Lot 29 : pull extra features from live_style if present
            extras = None
            if len(ctx["style_df"]) > 0:
                row = ctx["style_df"][ctx["style_df"]["fixture_id"] == fx_id]
                if len(row) > 0:
                    r = row.iloc[0].to_dict()
                    extras = {k: v for k, v in r.items()
                                if k not in ("fixture_id",) and v is not None}
            payload = ctx["enrich_fixture"](
                fixture_id=fx_id, feature_vector=vec,
                elo_diff=elo_diff, league_id=league_id,
                kickoff_utc=match.get("date"),
                extra_features=extras)
            pred = _prediction_dict_from_enrichment(payload)
            results.append({
                "fixture_id": match["fixture_id"],
                "league_id": match["league_id"],
                "league_name": match.get("league_name", ""),
                "league_flag": match.get("league_flag", ""),
                "league_country": match.get("league_country", ""),
                "date": match.get("date", ""),
                "venue": match.get("venue", ""),
                "match_status": match.get("match_status", "NS"),
                "match_elapsed": match.get("match_elapsed"),
                "current_home_goals": match.get("current_home_goals"),
                "current_away_goals": match.get("current_away_goals"),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "prediction": pred,
                "features": {"elo_diff": elo_diff},
                "odds_data": match.get("odds"),
                "odd_over15": match.get("odd_over15"),
                "odd_btts": match.get("odd_btts"),
                "odds_fetched_at": match.get("odds_fetched_at"),
                "referee_data": match.get("referee"),
                "lineups": match.get("lineups"),
                "injuries": match.get("injuries", []),
                "home_missing_players": match.get("home_missing_players", 0),
                "away_missing_players": match.get("away_missing_players", 0),
                "home_last_matches": match.get("home_last_matches", []),
                "away_last_matches": match.get("away_last_matches", []),
                "h2h": match.get("h2h", []),
            })
            to_persist.append(payload)
        except Exception as exc:
            log.warning("bridge_v2: fixture %s failed: %s", match.get("fixture_id"), exc)
            continue

    if persist and to_persist:
        try:
            ctx["append_enrichments"](to_persist)
        except Exception as exc:
            log.warning("bridge_v2: enrichment persistence failed: %s", exc)

    results.sort(key=lambda r: r["prediction"]["safety_score"], reverse=True)
    log.info("bridge_v2: %d predictions (persisted=%s)", len(results), persist)
    return results


def enrichment_ledger_path() -> Path:
    return _load_v2()["enrich_path"]


def load_enrichment_history() -> pd.DataFrame:
    p = enrichment_ledger_path()
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
