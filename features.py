"""
Smart Sim — Feature Engineering Avancé
Transforme les données brutes en vecteur de features pour le modèle Over 2.5.
"""

import math
import logging
from typing import Optional
from statistics import mean, stdev

import numpy as np
import pandas as pd

from config import LAST_N_MATCHES, STATS_KEYS, SMART_BET_THRESHOLD, EUROPEAN_CUP_IDS

log = logging.getLogger("SmartSim.features")


# ══════════════════════════════════════════════
# UTILITAIRES D'EXTRACTION
# ══════════════════════════════════════════════
def _safe(val, default=0.0) -> float:
    """Convertit une valeur en float, retourne default si impossible."""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip().rstrip("%")
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_team_stat(match: dict, team_name: str, stat_key: str) -> float:
    """Récupère une stat d'une équipe dans un match enrichi."""
    stats = match.get("match_stats") or {}
    team_stats = stats.get(team_name, {})
    return _safe(team_stats.get(stat_key))


def _get_goals(match: dict, team_id: int) -> tuple[int, int]:
    """Retourne (goals_scored, goals_conceded) pour une équipe dans un match."""
    home_id = match.get("teams", {}).get("home", {}).get("id")
    home_g = _safe(match.get("goals", {}).get("home"))
    away_g = _safe(match.get("goals", {}).get("away"))
    if team_id == home_id:
        return int(home_g), int(away_g)
    return int(away_g), int(home_g)


def _team_name_in_match(match: dict, team_id: int) -> str:
    """Retrouve le nom d'une équipe dans un match."""
    home = match.get("teams", {}).get("home", {})
    away = match.get("teams", {}).get("away", {})
    if home.get("id") == team_id:
        return home.get("name", "")
    return away.get("name", "")


# ══════════════════════════════════════════════
# FEATURES PAR ÉQUIPE (sur ses N derniers matchs)
# ══════════════════════════════════════════════
def compute_team_features(matches: list[dict], team_id: int, prefix: str) -> dict:
    """
    Calcule l'ensemble des features pour UNE équipe sur ses derniers matchs.
    prefix = 'home_' ou 'away_'
    """
    if not matches:
        return _empty_team_features(prefix)

    n = len(matches)
    # Poids exponentiels (0.5^i) — plus le match est récent, plus il pèse
    weights = [0.5 ** i for i in range(n)]
    w_sum = sum(weights) or 1.0

    # ── Buts ──
    goals_scored = []
    goals_conceded = []
    total_goals_match = []
    over25_count = 0

    # ── Stats agrégées ──
    xg_list = []
    shots_on_goal = []
    shots_inside_box = []
    total_shots = []
    possession_list = []
    pass_accuracy = []
    corners_list = []
    offsides_list = []
    gk_saves = []
    blocked_shots = []
    tackles_list = []
    interceptions_list = []
    fouls_list = []
    yellows_list = []
    reds_list = []

    # ── Dynamique temporelle ──
    first_goal_minutes = []
    goals_last_15_list = []

    # ── Splits domicile/extérieur (équipe jouait chez elle ou dehors ?) ──
    # prefix='home_' → on s'intéresse aux matchs OÙ ELLE A JOUÉ À DOMICILE
    # prefix='away_' → on s'intéresse aux matchs OÙ ELLE A JOUÉ À L'EXTÉRIEUR
    ctx_scored = []  # buts marqués dans le contexte du match à venir
    ctx_conceded = []
    ctx_btts = 0
    ctx_over25 = 0

    for match in matches:
        team_name = _team_name_in_match(match, team_id)
        scored, conceded = _get_goals(match, team_id)
        total = scored + conceded

        goals_scored.append(scored)
        goals_conceded.append(conceded)
        total_goals_match.append(total)
        if total > 2:
            over25_count += 1

        # Contexte : l'équipe jouait-elle home ou away dans CE match passé ?
        match_home_id = match.get("teams", {}).get("home", {}).get("id")
        was_at_home = (match_home_id == team_id)
        # On ne garde que les matchs joués dans le même contexte que le match à venir
        if (prefix == "home_" and was_at_home) or (prefix == "away_" and not was_at_home):
            ctx_scored.append(scored)
            ctx_conceded.append(conceded)
            if total > 2:
                ctx_over25 += 1
            if scored > 0 and conceded > 0:
                ctx_btts += 1

        # Stats du match
        xg_list.append(_get_team_stat(match, team_name, "expected_goals"))
        shots_on_goal.append(_get_team_stat(match, team_name, "Shots on Goal"))
        shots_inside_box.append(_get_team_stat(match, team_name, "Shots insidebox"))
        total_shots.append(_get_team_stat(match, team_name, "Total Shots"))
        possession_list.append(_get_team_stat(match, team_name, "Ball Possession"))
        pass_accuracy.append(_get_team_stat(match, team_name, "Passes %"))
        corners_list.append(_get_team_stat(match, team_name, "Corner Kicks"))
        offsides_list.append(_get_team_stat(match, team_name, "Offsides"))
        gk_saves.append(_get_team_stat(match, team_name, "Goalkeeper Saves"))
        blocked_shots.append(_get_team_stat(match, team_name, "Blocked Shots"))
        tackles_list.append(_get_team_stat(match, team_name, "Tackles"))
        interceptions_list.append(_get_team_stat(match, team_name, "Interceptions"))
        fouls_list.append(_get_team_stat(match, team_name, "Fouls"))
        yellows_list.append(_get_team_stat(match, team_name, "Yellow Cards"))
        reds_list.append(_get_team_stat(match, team_name, "Red Cards"))

        # Goal timings
        gt = match.get("goal_timings", {})
        if gt.get("first_goal_minute") is not None:
            first_goal_minutes.append(gt["first_goal_minute"])
        goals_last_15_list.append(gt.get("goals_last_15", 0))

    # ── Moyennes ──
    f = {}
    p = prefix

    # Buts
    f[f"{p}avg_scored"] = _avg(goals_scored)
    f[f"{p}avg_conceded"] = _avg(goals_conceded)
    f[f"{p}avg_total_goals"] = _avg(total_goals_match)
    f[f"{p}over25_rate"] = round(over25_count / n, 4) if n > 0 else 0.0

    # xG — avec fallback proxy si l'API ne fournit pas (formule Opta-like)
    real_xg_count = sum(1 for v in xg_list if v and v > 0.05)
    xg_is_real = 1 if real_xg_count >= max(2, n // 2) else 0
    if xg_is_real:
        f[f"{p}avg_xg"] = _avg(xg_list)
    else:
        # Proxy : 0.10 × tirs cadrés + 0.06 × tirs surface
        proxy_xg = [
            0.10 * _safe(shots_on_goal[i]) + 0.06 * _safe(shots_inside_box[i])
            for i in range(n)
        ]
        f[f"{p}avg_xg"] = _avg(proxy_xg)
    f[f"{p}xg_is_real"] = xg_is_real
    f[f"{p}xg_overperformance"] = round(f[f"{p}avg_scored"] - f[f"{p}avg_xg"], 4)

    # xG pondéré exponentiellement (forme récente prime)
    xg_for_weight = xg_list if xg_is_real else proxy_xg
    f[f"{p}weighted_avg_xg"] = round(
        sum(_safe(v) * w for v, w in zip(xg_for_weight, weights)) / w_sum, 4
    )

    # Offensive
    f[f"{p}avg_shots_on_goal"] = _avg(shots_on_goal)
    f[f"{p}avg_shots_inside_box"] = _avg(shots_inside_box)
    f[f"{p}avg_total_shots"] = _avg(total_shots)
    f[f"{p}shot_accuracy"] = (
        round(f[f"{p}avg_shots_on_goal"] / f[f"{p}avg_total_shots"], 4)
        if f[f"{p}avg_total_shots"] > 0 else 0.0
    )

    # Possession & Pression
    f[f"{p}avg_possession"] = _avg(possession_list)
    f[f"{p}avg_pass_accuracy"] = _avg(pass_accuracy)
    f[f"{p}avg_corners"] = _avg(corners_list)
    f[f"{p}avg_offsides"] = _avg(offsides_list)

    # Défense
    f[f"{p}avg_gk_saves"] = _avg(gk_saves)
    f[f"{p}avg_blocked_shots"] = _avg(blocked_shots)
    f[f"{p}avg_tackles"] = _avg(tackles_list)
    f[f"{p}avg_interceptions"] = _avg(interceptions_list)

    # Discipline
    f[f"{p}avg_fouls"] = _avg(fouls_list)
    f[f"{p}avg_yellows"] = _avg(yellows_list)
    f[f"{p}avg_reds"] = _avg(reds_list)

    # ── INDICATEURS COMPOSITES ──

    # 1. Dangerous Possession Index = (Corners + Tirs cadrés) / Possession
    avg_poss = f[f"{p}avg_possession"]
    f[f"{p}dangerous_possession_idx"] = (
        round((f[f"{p}avg_corners"] + f[f"{p}avg_shots_on_goal"]) / avg_poss, 4)
        if avg_poss > 0 else 0.0
    )

    # 2. Defense-to-Attack Transition = Interceptions / Tirs tentés
    avg_shots = f[f"{p}avg_total_shots"]
    f[f"{p}defense_attack_transition"] = (
        round(f[f"{p}avg_interceptions"] / avg_shots, 4)
        if avg_shots > 0 else 0.0
    )

    # 3. Chaos Factor = écart-type des buts marqués sur les 5 derniers matchs
    recent_scored = goals_scored[:5]
    f[f"{p}chaos_factor"] = round(stdev(recent_scored), 4) if len(recent_scored) >= 2 else 0.0

    # 4. Goal Volatility = écart-type du total de buts sur les 5 derniers matchs
    recent_total = total_goals_match[:5]
    f[f"{p}goal_volatility"] = round(stdev(recent_total), 4) if len(recent_total) >= 2 else 0.0

    # 5. Dynamique temporelle
    f[f"{p}avg_first_goal_minute"] = _avg(first_goal_minutes) if first_goal_minutes else 45.0
    f[f"{p}avg_goals_last_15"] = _avg(goals_last_15_list)

    # 6. Scoring Consistency = min(goals_scored[-5:]) / max(goals_scored[-5:])
    if recent_scored and max(recent_scored) > 0:
        f[f"{p}scoring_consistency"] = round(min(recent_scored) / max(recent_scored), 4)
    else:
        f[f"{p}scoring_consistency"] = 0.0

    # 7. Conversion Rate = buts / tirs cadrés
    f[f"{p}conversion_rate"] = (
        round(f[f"{p}avg_scored"] / f[f"{p}avg_shots_on_goal"], 4)
        if f[f"{p}avg_shots_on_goal"] > 0 else 0.0
    )

    # 8. Defensive Pressure = (Tacles + Interceptions) / Possession adverse
    f[f"{p}defensive_pressure"] = (
        round((f[f"{p}avg_tackles"] + f[f"{p}avg_interceptions"]) / (100 - avg_poss), 4)
        if avg_poss < 100 else 0.0
    )

    # 9. Clean Sheet Rate
    f[f"{p}clean_sheet_rate"] = round(
        sum(1 for g in goals_conceded if g == 0) / n, 4
    ) if n > 0 else 0.0

    # 10. BTTS Rate (Both Teams To Score)
    btts = sum(1 for i in range(n) if goals_scored[i] > 0 and goals_conceded[i] > 0)
    f[f"{p}btts_rate"] = round(btts / n, 4) if n > 0 else 0.0

    # 11. Form (weighted recent — poids exponentiels : 0.5^i)
    f[f"{p}weighted_avg_scored"] = round(
        sum(g * w for g, w in zip(goals_scored, weights)) / w_sum, 4
    )
    f[f"{p}weighted_avg_conceded"] = round(
        sum(g * w for g, w in zip(goals_conceded, weights)) / w_sum, 4
    )
    # Pondération exponentielle étendue à shots, possession (forme récente)
    f[f"{p}weighted_avg_shots_on"] = round(
        sum(_safe(v) * w for v, w in zip(shots_on_goal, weights)) / w_sum, 4
    )
    f[f"{p}weighted_avg_possession"] = round(
        sum(_safe(v) * w for v, w in zip(possession_list, weights)) / w_sum, 4
    )

    # ── Splits contextuels (équipe dans son contexte de match à venir) ──
    # ctx_* = stats filtrées sur les matchs joués dans le bon contexte (home OU away)
    n_ctx = len(ctx_scored)
    if n_ctx > 0:
        f[f"{p}ctx_avg_scored"] = round(sum(ctx_scored) / n_ctx, 4)
        f[f"{p}ctx_avg_conceded"] = round(sum(ctx_conceded) / n_ctx, 4)
        f[f"{p}ctx_over25_rate"] = round(ctx_over25 / n_ctx, 4)
        f[f"{p}ctx_btts_rate"] = round(ctx_btts / n_ctx, 4)
        f[f"{p}ctx_sample_size"] = n_ctx
    else:
        # Pas de matchs dans le bon contexte → fallback sur les stats globales
        f[f"{p}ctx_avg_scored"] = _avg(goals_scored)
        f[f"{p}ctx_avg_conceded"] = _avg(goals_conceded)
        f[f"{p}ctx_over25_rate"] = round(over25_count / n, 4) if n > 0 else 0.0
        f[f"{p}ctx_btts_rate"] = 0.0
        f[f"{p}ctx_sample_size"] = 0

    # 12. Forme ultra-récente (3 derniers matchs)
    last3_scored = goals_scored[:3]
    last3_conceded = goals_conceded[:3]
    last3_total = total_goals_match[:3]
    f[f"{p}form_last3_scored"] = _avg(last3_scored)
    f[f"{p}form_last3_conceded"] = _avg(last3_conceded)
    f[f"{p}form_last3_over25_rate"] = round(
        sum(1 for t in last3_total if t > 2) / len(last3_total), 4
    ) if last3_total else 0.0

    return f


def _empty_team_features(prefix: str) -> dict:
    """Retourne un dict de features vides pour une équipe sans données."""
    keys = [
        "avg_scored", "avg_conceded", "avg_total_goals", "over25_rate",
        "avg_xg", "xg_overperformance",
        "avg_shots_on_goal", "avg_shots_inside_box", "avg_total_shots", "shot_accuracy",
        "avg_possession", "avg_pass_accuracy", "avg_corners", "avg_offsides",
        "avg_gk_saves", "avg_blocked_shots", "avg_tackles", "avg_interceptions",
        "avg_fouls", "avg_yellows", "avg_reds",
        "dangerous_possession_idx", "defense_attack_transition",
        "chaos_factor", "goal_volatility",
        "avg_first_goal_minute", "avg_goals_last_15",
        "scoring_consistency", "conversion_rate", "defensive_pressure",
        "clean_sheet_rate", "btts_rate",
        "weighted_avg_scored", "weighted_avg_conceded",
        "weighted_avg_xg", "weighted_avg_shots_on", "weighted_avg_possession",
        "xg_is_real",
        "form_last3_scored", "form_last3_conceded", "form_last3_over25_rate",
        "ctx_avg_scored", "ctx_avg_conceded", "ctx_over25_rate", "ctx_btts_rate",
        "ctx_sample_size",
    ]
    return {f"{prefix}{k}": 0.0 for k in keys}


# ══════════════════════════════════════════════
# FEATURES H2H
# ══════════════════════════════════════════════
def compute_h2h_features(h2h_matches: list[dict], home_id: int, away_id: int) -> dict:
    """Features des confrontations directes."""
    f = {}
    n = len(h2h_matches)

    if n == 0:
        return {
            "h2h_count": 0,
            "h2h_avg_total_goals": 0.0,
            "h2h_over25_rate": 0.0,
            "h2h_btts_rate": 0.0,
            "h2h_home_win_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_avg_first_goal_min": 45.0,
            "h2h_avg_goals_last_15": 0.0,
        }

    total_goals = []
    over25 = 0
    btts = 0
    home_wins = 0
    away_wins = 0
    draws = 0
    first_goal_mins = []
    goals_last_15 = []

    for match in h2h_matches:
        hg = _safe(match.get("goals", {}).get("home"))
        ag = _safe(match.get("goals", {}).get("away"))
        total = int(hg) + int(ag)
        total_goals.append(total)

        if total > 2:
            over25 += 1
        if hg > 0 and ag > 0:
            btts += 1

        # Résultat du H2H du point de vue de home_id (équipe domicile d'aujourd'hui)
        match_home_id = match.get("teams", {}).get("home", {}).get("id")
        if hg == ag:
            draws += 1
        elif match_home_id == home_id:
            # home_id jouait à domicile dans ce H2H
            if hg > ag:
                home_wins += 1
            else:
                away_wins += 1
        else:
            # home_id jouait à l'extérieur dans ce H2H
            if ag > hg:
                home_wins += 1
            else:
                away_wins += 1

        gt = match.get("goal_timings", {})
        if gt.get("first_goal_minute") is not None:
            first_goal_mins.append(gt["first_goal_minute"])
        goals_last_15.append(gt.get("goals_last_15", 0))

    f["h2h_count"] = n
    f["h2h_avg_total_goals"] = _avg(total_goals)
    f["h2h_over25_rate"] = round(over25 / n, 4)
    f["h2h_btts_rate"] = round(btts / n, 4)
    f["h2h_home_win_rate"] = round(home_wins / n, 4)
    f["h2h_away_win_rate"] = round(away_wins / n, 4)
    f["h2h_draw_rate"] = round(draws / n, 4)
    f["h2h_avg_first_goal_min"] = _avg(first_goal_mins) if first_goal_mins else 45.0
    f["h2h_avg_goals_last_15"] = _avg(goals_last_15)

    return f


# ══════════════════════════════════════════════
# FEATURES ARBITRE
# ══════════════════════════════════════════════
def compute_referee_features(referee_data: Optional[dict]) -> dict:
    """Features liées à l'arbitre."""
    if not referee_data:
        return {
            "ref_avg_total_goals": 0.0,
            "ref_over25_rate": 0.0,
            "ref_penalty_rate": 0.0,
            "ref_avg_yellows": 0.0,
            "ref_avg_fouls": 0.0,
            "ref_has_data": 0,
        }

    return {
        "ref_avg_total_goals": _safe(referee_data.get("avg_total_goals")),
        "ref_over25_rate": _safe(referee_data.get("over_25_rate")),
        "ref_penalty_rate": _safe(referee_data.get("penalty_rate")),
        "ref_avg_yellows": _safe(referee_data.get("avg_yellows_per_match")),
        "ref_avg_fouls": _safe(referee_data.get("avg_fouls_per_match")),
        "ref_has_data": 1,
    }


# ══════════════════════════════════════════════
# FEATURES COTES / MARCHÉ
# ══════════════════════════════════════════════
def compute_odds_features(market_data: Optional[dict]) -> dict:
    """Features dérivées des cotes bookmakers."""
    if not market_data:
        return {
            "odds_avg_over25": 0.0,
            "odds_implied_prob_over": 0.0,
            "odds_market_over": 0,
            "odds_spread": 0.0,
            "odds_has_data": 0,
        }

    return {
        "odds_avg_over25": _safe(market_data.get("avg_over_25")),
        "odds_implied_prob_over": _safe(market_data.get("implied_prob_over")),
        "odds_market_over": 1 if market_data.get("market_direction") == "over" else 0,
        "odds_spread": _safe(market_data.get("spread")),
        "odds_has_data": 1,
    }


def compute_odds_extra_features(odd_over15, odd_btts) -> dict:
    """Cotes Over 1.5 et BTTS converties en proba implicite (1/cote)."""
    f = {
        "odds_implied_prob_o15": 0.0,
        "odds_o15_has_data": 0,
        "odds_implied_prob_btts": 0.0,
        "odds_btts_has_data": 0,
    }
    o15 = _safe(odd_over15)
    if o15 and o15 > 1.0:
        f["odds_implied_prob_o15"] = round(1.0 / o15, 4)
        f["odds_o15_has_data"] = 1
    btts = _safe(odd_btts)
    if btts and btts > 1.0:
        f["odds_implied_prob_btts"] = round(1.0 / btts, 4)
        f["odds_btts_has_data"] = 1
    return f


# ══════════════════════════════════════════════
# FEATURES CLASSEMENT (standings de la ligue)
# ══════════════════════════════════════════════
def compute_standings_features(home_standing: Optional[dict],
                                away_standing: Optional[dict],
                                league_meta: Optional[dict] = None) -> dict:
    """
    Features de classement : rank, points par match (overall + contextuels).
    Le contexte = home joue à domicile, away joue à l'extérieur.
    league_meta : dict avec top_points, safety_points, max_played, total_teams
                  pour les features de motivation contextuelle.
    """
    f = {
        "home_rank": 0.0, "away_rank": 0.0, "rank_diff": 0.0,
        "home_ppg_overall": 0.0, "away_ppg_overall": 0.0,
        "home_ppg_at_home": 0.0, "away_ppg_at_away": 0.0,
        "home_gpg_at_home": 0.0, "away_gpg_at_away": 0.0,
        "home_gapg_at_home": 0.0, "away_gapg_at_away": 0.0,
        "standings_has_data": 0,
        # Motivation contextuelle
        "matchday_pct": 0.0,
        "home_pts_to_safety": 0.0, "away_pts_to_safety": 0.0,
        "home_pts_to_top": 0.0, "away_pts_to_top": 0.0,
        "home_dead_rubber": 0, "away_dead_rubber": 0,
        "match_has_stakes": 0,
    }
    if not home_standing or not away_standing:
        return f

    f["home_rank"] = _safe(home_standing.get("rank"))
    f["away_rank"] = _safe(away_standing.get("rank"))
    # rank_diff < 0 → home mieux classé (avantage home)
    f["rank_diff"] = f["home_rank"] - f["away_rank"]

    h_played = max(1, _safe(home_standing.get("played")))
    a_played = max(1, _safe(away_standing.get("played")))
    f["home_ppg_overall"] = round(_safe(home_standing.get("points")) / h_played, 4)
    f["away_ppg_overall"] = round(_safe(away_standing.get("points")) / a_played, 4)

    # PPG dans le contexte du match (home à domicile, away à l'extérieur)
    h_home_played = (_safe(home_standing.get("home_wins"))
                     + _safe(home_standing.get("home_draws"))
                     + _safe(home_standing.get("home_losses")))
    if h_home_played > 0:
        h_home_pts = 3 * _safe(home_standing.get("home_wins")) + _safe(home_standing.get("home_draws"))
        f["home_ppg_at_home"] = round(h_home_pts / h_home_played, 4)
        f["home_gpg_at_home"] = round(_safe(home_standing.get("home_goals_for")) / h_home_played, 4)
        f["home_gapg_at_home"] = round(_safe(home_standing.get("home_goals_against")) / h_home_played, 4)

    a_away_played = (_safe(away_standing.get("away_wins"))
                     + _safe(away_standing.get("away_draws"))
                     + _safe(away_standing.get("away_losses")))
    if a_away_played > 0:
        a_away_pts = 3 * _safe(away_standing.get("away_wins")) + _safe(away_standing.get("away_draws"))
        f["away_ppg_at_away"] = round(a_away_pts / a_away_played, 4)
        f["away_gpg_at_away"] = round(_safe(away_standing.get("away_goals_for")) / a_away_played, 4)
        f["away_gapg_at_away"] = round(_safe(away_standing.get("away_goals_against")) / a_away_played, 4)

    f["standings_has_data"] = 1

    # ── Motivation contextuelle ──
    if league_meta:
        max_played = _safe(league_meta.get("max_played"))
        # Saison standard ≈ 38 journées en championnat (2 × n_teams - 2 pour aller-retour)
        n_teams = _safe(league_meta.get("total_teams")) or 20
        total_matchdays = max(10, 2 * (n_teams - 1))
        if max_played > 0:
            f["matchday_pct"] = round(min(1.0, max_played / total_matchdays), 4)

        top_pts = _safe(league_meta.get("top_points"))
        safety_pts = _safe(league_meta.get("safety_points"))
        h_pts = _safe(home_standing.get("points"))
        a_pts = _safe(away_standing.get("points"))
        f["home_pts_to_safety"] = round(h_pts - safety_pts, 2)
        f["away_pts_to_safety"] = round(a_pts - safety_pts, 2)
        f["home_pts_to_top"] = round(top_pts - h_pts, 2)
        f["away_pts_to_top"] = round(top_pts - a_pts, 2)

        # Dead rubber : fin de saison + équipe ni dans la course au titre ni en danger
        is_late_season = f["matchday_pct"] > 0.80
        f["home_dead_rubber"] = int(
            is_late_season and f["home_pts_to_safety"] > 10 and f["home_pts_to_top"] > 15
        )
        f["away_dead_rubber"] = int(
            is_late_season and f["away_pts_to_safety"] > 10 and f["away_pts_to_top"] > 15
        )
        # Le match a des enjeux si AU MOINS UNE équipe joue qqchose
        f["match_has_stakes"] = int(
            not f["home_dead_rubber"] or not f["away_dead_rubber"]
        )

    return f


# ══════════════════════════════════════════════
# FEATURES CONTEXTUELLES
# ══════════════════════════════════════════════
def compute_context_features(match_data: dict) -> dict:
    """Features contextuelles : repos, domicile/extérieur…"""
    home_rest = match_data.get("home_rest_days")
    away_rest = match_data.get("away_rest_days")

    f = {}

    # Fatigue Score (jours de repos)
    f["home_fatigue_score"] = _fatigue_score(home_rest)
    f["away_fatigue_score"] = _fatigue_score(away_rest)
    f["home_rest_days"] = _safe(home_rest, default=4.0)
    f["away_rest_days"] = _safe(away_rest, default=4.0)
    f["rest_diff"] = f["home_rest_days"] - f["away_rest_days"]

    return f


def _fatigue_score(rest_days: Optional[int]) -> float:
    """
    Score de fatigue basé sur les jours de repos.
    0 = reposé, 1 = très fatigué.
    Optimal = 4-6 jours. < 3 ou > 10 = pénalité.
    """
    if rest_days is None:
        return 0.3  # valeur neutre
    if rest_days <= 2:
        return 0.9
    if rest_days == 3:
        return 0.6
    if 4 <= rest_days <= 6:
        return 0.1
    if 7 <= rest_days <= 10:
        return 0.2
    return 0.4  # > 10 jours = manque de rythme


# ══════════════════════════════════════════════
# FEATURES CROISÉES (interaction home × away)
# ══════════════════════════════════════════════
def compute_cross_features(features: dict) -> dict:
    """Features d'interaction entre les deux équipes."""
    f = {}

    # Somme des moyennes de buts
    f["combined_avg_scored"] = features.get("home_avg_scored", 0) + features.get("away_avg_scored", 0)
    f["combined_avg_conceded"] = features.get("home_avg_conceded", 0) + features.get("away_avg_conceded", 0)
    f["combined_avg_total"] = features.get("home_avg_total_goals", 0) + features.get("away_avg_total_goals", 0)

    # xG combiné
    f["combined_xg"] = features.get("home_avg_xg", 0) + features.get("away_avg_xg", 0)

    # Over 2.5 rate combinée (moyenne)
    f["combined_over25_rate"] = (
        features.get("home_over25_rate", 0) + features.get("away_over25_rate", 0)
    ) / 2

    # BTTS combiné
    f["combined_btts_rate"] = (
        features.get("home_btts_rate", 0) + features.get("away_btts_rate", 0)
    ) / 2

    # Différence de puissance offensive
    f["attack_diff"] = features.get("home_avg_scored", 0) - features.get("away_avg_scored", 0)

    # Match Openness = attaque home vs défense away + attaque away vs défense home
    f["match_openness"] = (
        features.get("home_avg_scored", 0) + features.get("away_avg_conceded", 0) +
        features.get("away_avg_scored", 0) + features.get("home_avg_conceded", 0)
    ) / 2

    # Chaos combiné
    f["combined_chaos"] = (
        features.get("home_chaos_factor", 0) + features.get("away_chaos_factor", 0)
    ) / 2

    # Dangerous Possession combiné
    f["combined_dangerous_poss"] = (
        features.get("home_dangerous_possession_idx", 0) +
        features.get("away_dangerous_possession_idx", 0)
    )

    # Defense Weakness = somme des buts encaissés moyens
    f["defense_weakness"] = (
        features.get("home_avg_conceded", 0) + features.get("away_avg_conceded", 0)
    )

    # xG vs Reality gap (sur-performance ou sous-performance combinée)
    f["combined_xg_gap"] = (
        features.get("home_xg_overperformance", 0) + features.get("away_xg_overperformance", 0)
    )

    # Late Goals Trend
    f["combined_goals_last_15"] = (
        features.get("home_avg_goals_last_15", 0) + features.get("away_avg_goals_last_15", 0)
    )

    # Early Goal Likelihood (plus le 1er but tombe tôt, plus de temps pour des buts)
    avg_first_h = features.get("home_avg_first_goal_minute", 45)
    avg_first_a = features.get("away_avg_first_goal_minute", 45)
    f["avg_first_goal_combined"] = (avg_first_h + avg_first_a) / 2
    f["early_goal_score"] = max(0, 1 - f["avg_first_goal_combined"] / 90)

    return f


# ══════════════════════════════════════════════
# FEATURES COUPES D'EUROPE (double analyse)
# ══════════════════════════════════════════════
def compute_euro_cup_features(euro_history: list[dict], team_id: int, prefix: str) -> dict:
    """
    Pilier 1 : Data de Compétition (le socle).
    Analyse l'historique d'une équipe DANS la coupe européenne spécifique
    (LDC, UEL, UECL) sur les dernières saisons.
    Capture l'expérience européenne et le niveau de performance en coupe.
    """
    f = {}
    base = f"{prefix}euro_"

    if not euro_history:
        # Pas d'historique européen = équipe novice
        f[f"{base}matches_played"] = 0
        f[f"{base}avg_scored"] = 0.0
        f[f"{base}avg_conceded"] = 0.0
        f[f"{base}avg_total_goals"] = 0.0
        f[f"{base}over25_rate"] = 0.0
        f[f"{base}win_rate"] = 0.0
        f[f"{base}clean_sheet_rate"] = 0.0
        f[f"{base}experience_score"] = 0.0
        return f

    goals_scored = []
    goals_conceded = []
    wins = 0
    clean_sheets = 0

    for match in euro_history:
        home_id = match.get("teams", {}).get("home", {}).get("id")
        home_g = _safe(match.get("goals", {}).get("home"))
        away_g = _safe(match.get("goals", {}).get("away"))

        if team_id == home_id:
            scored, conceded = int(home_g), int(away_g)
        else:
            scored, conceded = int(away_g), int(home_g)

        goals_scored.append(scored)
        goals_conceded.append(conceded)
        if scored > conceded:
            wins += 1
        if conceded == 0:
            clean_sheets += 1

    n = len(euro_history)
    total_goals = [s + c for s, c in zip(goals_scored, goals_conceded)]

    f[f"{base}matches_played"] = n
    f[f"{base}avg_scored"] = round(mean(goals_scored), 4) if goals_scored else 0.0
    f[f"{base}avg_conceded"] = round(mean(goals_conceded), 4) if goals_conceded else 0.0
    f[f"{base}avg_total_goals"] = round(mean(total_goals), 4) if total_goals else 0.0
    f[f"{base}over25_rate"] = round(sum(1 for t in total_goals if t > 2.5) / n, 4) if n else 0.0
    f[f"{base}win_rate"] = round(wins / n, 4) if n else 0.0
    f[f"{base}clean_sheet_rate"] = round(clean_sheets / n, 4) if n else 0.0
    # Score d'expérience : 0→1, saturé à 15 matchs européens
    f[f"{base}experience_score"] = round(min(n / 15, 1.0), 4)

    return f


def compute_euro_weighting_features(form_features: dict, euro_features: dict,
                                     prefix: str) -> dict:
    """
    Pondération croisée forme x compétition.
    Si une équipe survole son championnat MAIS est médiocre en coupe européenne,
    cela doit se refléter dans le score final.
    """
    f = {}
    base = f"{prefix}euro_"

    form_win_proxy = form_features.get(f"{prefix}over25_rate", 0.5)
    euro_over25 = euro_features.get(f"{base}over25_rate", 0.5)
    euro_exp = euro_features.get(f"{base}experience_score", 0.0)

    # Divergence forme championnat vs historique coupe
    f[f"{base}form_vs_cup_gap"] = round(form_win_proxy - euro_over25, 4)

    # Score pondéré : mix forme actuelle (60%) + historique coupe (40%)
    # Pondéré par l'expérience : si pas d'expérience, on se fie plus à la forme
    cup_weight = 0.4 * euro_exp  # 0 à 0.4 selon expérience
    form_weight = 1.0 - cup_weight

    form_scored = form_features.get(f"{prefix}avg_scored", 0.0)
    euro_scored = euro_features.get(f"{base}avg_scored", 0.0)
    f[f"{base}weighted_attack"] = round(
        form_scored * form_weight + euro_scored * cup_weight, 4
    )

    form_conceded = form_features.get(f"{prefix}avg_conceded", 0.0)
    euro_conceded = euro_features.get(f"{base}avg_conceded", 0.0)
    f[f"{base}weighted_defense"] = round(
        form_conceded * form_weight + euro_conceded * cup_weight, 4
    )

    return f


# ══════════════════════════════════════════════
# PIPELINE COMPLET : MATCH → FEATURE VECTOR
# ══════════════════════════════════════════════
def build_feature_vector(match_data: dict) -> Optional[dict]:
    """
    Transforme un match enrichi (sortie de data_fetcher.fetch_full_match_data)
    en un vecteur de features prêt pour le modèle.

    Retourne un dict plat {feature_name: value} ou None si données insuffisantes.
    """
    home_id = match_data["home_team"]["id"]
    away_id = match_data["away_team"]["id"]

    # ① Features par équipe
    home_features = compute_team_features(
        match_data.get("home_last_matches", []), home_id, "home_"
    )
    away_features = compute_team_features(
        match_data.get("away_last_matches", []), away_id, "away_"
    )

    # ② Features H2H
    h2h_features = compute_h2h_features(
        match_data.get("h2h", []), home_id, away_id
    )

    # ③ Features Arbitre
    referee_features = compute_referee_features(match_data.get("referee"))

    # ④ Features Cotes
    odds_features = compute_odds_features(match_data.get("odds"))
    odds_extra = compute_odds_extra_features(
        match_data.get("odd_over15"), match_data.get("odd_btts")
    )

    # ⑤ Features Contextuelles
    context_features = compute_context_features(match_data)

    # ⑤bis Features Classement de ligue + motivation contextuelle
    standings_features = compute_standings_features(
        match_data.get("home_standing"),
        match_data.get("away_standing"),
        match_data.get("league_meta"),
    )

    # Merge all
    features = {}
    features.update(home_features)
    features.update(away_features)
    features.update(h2h_features)
    features.update(referee_features)
    features.update(odds_features)
    features.update(odds_extra)
    features.update(context_features)
    features.update(standings_features)

    # ⑥ Features croisées (nécessite les features précédentes)
    cross = compute_cross_features(features)
    features.update(cross)

    # ⑦ Features Coupes d'Europe (double analyse compétition + forme)
    is_euro = match_data.get("is_european_cup", False)
    home_euro = compute_euro_cup_features(
        match_data.get("home_euro_history", []), home_id, "home_"
    )
    away_euro = compute_euro_cup_features(
        match_data.get("away_euro_history", []), away_id, "away_"
    )
    features.update(home_euro)
    features.update(away_euro)

    if is_euro:
        home_weight = compute_euro_weighting_features(home_features, home_euro, "home_")
        away_weight = compute_euro_weighting_features(away_features, away_euro, "away_")
        features.update(home_weight)
        features.update(away_weight)
    else:
        # Zéros pour les features de pondération hors coupes d'Europe
        for prefix in ("home_", "away_"):
            base = f"{prefix}euro_"
            features[f"{base}form_vs_cup_gap"] = 0.0
            features[f"{base}weighted_attack"] = 0.0
            features[f"{base}weighted_defense"] = 0.0

    # ⑧ Joueurs absents / blessés (influence directe sur l'xG attendu)
    home_missing = _safe(match_data.get("home_missing_players", 0))
    away_missing = _safe(match_data.get("away_missing_players", 0))
    features["home_missing_count"] = home_missing
    features["away_missing_count"] = away_missing
    features["home_missing_impact"] = round(min(home_missing / 5.0, 1.0), 4)
    features["away_missing_impact"] = round(min(away_missing / 5.0, 1.0), 4)

    # Pondération avancée — basée sur le statut dans les lineups (titulaire/sub/absent)
    home_missing_w = _safe(match_data.get("home_missing_weighted", 0))
    away_missing_w = _safe(match_data.get("away_missing_weighted", 0))
    features["home_missing_weighted"] = round(home_missing_w, 4)
    features["away_missing_weighted"] = round(away_missing_w, 4)
    features["home_missing_weighted_impact"] = round(min(home_missing_w / 4.0, 1.0), 4)
    features["away_missing_weighted_impact"] = round(min(away_missing_w / 4.0, 1.0), 4)

    # Formation offensive/défensive (1=offensif, 0=neutre, -1=défensif)
    features["home_formation_offensive"] = int(match_data.get("home_formation_score", 0) or 0)
    features["away_formation_offensive"] = int(match_data.get("away_formation_score", 0) or 0)

    return features


def build_features_dataframe(matches_data: list[dict]) -> pd.DataFrame:
    """
    Construit un DataFrame de features pour une liste de matchs.
    Chaque ligne = un match, chaque colonne = une feature.
    """
    rows = []
    metadata = []

    for match in matches_data:
        vec = build_feature_vector(match)
        if vec is not None:
            rows.append(vec)
            metadata.append({
                "fixture_id": match["fixture_id"],
                "league_id": match["league_id"],
                "league_name": match.get("league_name", ""),
                "date": match.get("date", ""),
                "home_team": match["home_team"]["name"],
                "away_team": match["away_team"]["name"],
                "home_logo": match["home_team"].get("logo", ""),
                "away_logo": match["away_team"].get("logo", ""),
                "venue": match.get("venue", ""),
                "odds_data": match.get("odds"),
                "referee_data": match.get("referee"),
            })

    if not rows:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)

    # Remplacer NaN / inf par 0
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df, metadata


# ══════════════════════════════════════════════
# SMART BET DETECTOR
# ══════════════════════════════════════════════
def detect_smart_bet(prediction_proba: float, features: dict) -> dict:
    """
    Détecte si un match est un 'SMART BET' :
    convergence entre IA, xG et marché.

    Retourne un dict avec le statut et les signaux convergents.
    """
    signals = []

    # Signal 1 : Modèle IA confiant
    if prediction_proba >= SMART_BET_THRESHOLD:
        signals.append("IA_HIGH")

    # Signal 2 : xG combiné élevé (> 2.5)
    combined_xg = features.get("combined_xg", 0)
    if combined_xg > 2.5:
        signals.append("XG_OVER")

    # Signal 3 : Marché penche Over
    if features.get("odds_market_over") == 1:
        signals.append("MARKET_OVER")

    # Signal 4 : H2H historiquement Over
    if features.get("h2h_over25_rate", 0) >= 0.6:
        signals.append("H2H_OVER")

    # Signal 5 : Les deux équipes marquent souvent
    if features.get("combined_btts_rate", 0) >= 0.65:
        signals.append("BTTS_HIGH")

    # Signal 6 : Arbitre permissif
    if features.get("ref_over25_rate", 0) >= 0.55:
        signals.append("REF_OVER")

    # Signal 7 : Buts tardifs fréquents
    if features.get("combined_goals_last_15", 0) >= 0.8:
        signals.append("LATE_GOALS")

    # SMART BET = au moins 3 signaux ET IA confiante
    is_smart = len(signals) >= 3 and "IA_HIGH" in signals

    # Confidence boost : plus de signaux = plus confiant
    convergence_score = round(len(signals) / 7, 4)

    return {
        "is_smart_bet": is_smart,
        "signals": signals,
        "signal_count": len(signals),
        "convergence_score": convergence_score,
    }


# ══════════════════════════════════════════════
# FEATURE IMPORTANCE (pour l'UI)
# ══════════════════════════════════════════════
def get_top_drivers(features: dict, feature_importances: dict, top_n: int = 5) -> list[dict]:
    """
    Identifie les top N features qui poussent le plus vers Over 2.5 pour ce match.
    Utile pour l'explication dans l'UI.
    """
    drivers = []
    for feat_name, importance in feature_importances.items():
        value = features.get(feat_name, 0)
        drivers.append({
            "feature": feat_name,
            "value": round(value, 4),
            "importance": round(importance, 4),
            "impact": round(value * importance, 4),
        })

    drivers.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return drivers[:top_n]


# ══════════════════════════════════════════════
# COLONNES DU MODÈLE (ordre garanti)
# ══════════════════════════════════════════════
def get_model_columns() -> list[str]:
    """
    Retourne la liste ordonnée des colonnes utilisées par le modèle.
    Garantit la cohérence entre entraînement et prédiction.
    """
    prefixes = ["home_", "away_"]
    team_features = [
        "avg_scored", "avg_conceded", "avg_total_goals", "over25_rate",
        "avg_xg", "xg_overperformance",
        "avg_shots_on_goal", "avg_shots_inside_box", "avg_total_shots", "shot_accuracy",
        "avg_possession", "avg_pass_accuracy", "avg_corners", "avg_offsides",
        "avg_gk_saves", "avg_blocked_shots", "avg_tackles", "avg_interceptions",
        "avg_fouls", "avg_yellows", "avg_reds",
        "dangerous_possession_idx", "defense_attack_transition",
        "chaos_factor", "goal_volatility",
        "avg_first_goal_minute", "avg_goals_last_15",
        "scoring_consistency", "conversion_rate", "defensive_pressure",
        "clean_sheet_rate", "btts_rate",
        "weighted_avg_scored", "weighted_avg_conceded",
        "weighted_avg_xg", "weighted_avg_shots_on", "weighted_avg_possession",
        "xg_is_real",
        # Forme ultra-récente (3 derniers matchs)
        "form_last3_scored", "form_last3_conceded", "form_last3_over25_rate",
        # Splits contextuels (home à domicile, away à l'extérieur)
        "ctx_avg_scored", "ctx_avg_conceded", "ctx_over25_rate", "ctx_btts_rate",
        "ctx_sample_size",
    ]

    cols = []
    for p in prefixes:
        for tf in team_features:
            cols.append(f"{p}{tf}")

    # H2H
    cols += [
        "h2h_count", "h2h_avg_total_goals", "h2h_over25_rate", "h2h_btts_rate",
        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
        "h2h_avg_first_goal_min", "h2h_avg_goals_last_15",
    ]

    # Arbitre
    cols += [
        "ref_avg_total_goals", "ref_over25_rate", "ref_penalty_rate",
        "ref_avg_yellows", "ref_avg_fouls", "ref_has_data",
    ]

    # Cotes
    cols += [
        "odds_avg_over25", "odds_implied_prob_over", "odds_market_over",
        "odds_spread", "odds_has_data",
        "odds_implied_prob_o15", "odds_o15_has_data",
        "odds_implied_prob_btts", "odds_btts_has_data",
    ]

    # Classement de ligue + motivation
    cols += [
        "home_rank", "away_rank", "rank_diff",
        "home_ppg_overall", "away_ppg_overall",
        "home_ppg_at_home", "away_ppg_at_away",
        "home_gpg_at_home", "away_gpg_at_away",
        "home_gapg_at_home", "away_gapg_at_away",
        "standings_has_data",
        "matchday_pct",
        "home_pts_to_safety", "away_pts_to_safety",
        "home_pts_to_top", "away_pts_to_top",
        "home_dead_rubber", "away_dead_rubber", "match_has_stakes",
    ]

    # Contexte
    cols += [
        "home_fatigue_score", "away_fatigue_score",
        "home_rest_days", "away_rest_days", "rest_diff",
    ]

    # Cross features
    cols += [
        "combined_avg_scored", "combined_avg_conceded", "combined_avg_total",
        "combined_xg", "combined_over25_rate", "combined_btts_rate",
        "attack_diff", "match_openness", "combined_chaos",
        "combined_dangerous_poss", "defense_weakness", "combined_xg_gap",
        "combined_goals_last_15", "avg_first_goal_combined", "early_goal_score",
    ]

    # Absents / blessés
    cols += [
        "home_missing_count", "away_missing_count",
        "home_missing_impact", "away_missing_impact",
        "home_missing_weighted", "away_missing_weighted",
        "home_missing_weighted_impact", "away_missing_weighted_impact",
        "home_formation_offensive", "away_formation_offensive",
    ]

    # Coupes d'Europe — historique compétition + pondération
    euro_features = [
        "euro_matches_played", "euro_avg_scored", "euro_avg_conceded",
        "euro_avg_total_goals", "euro_over25_rate", "euro_win_rate",
        "euro_clean_sheet_rate", "euro_experience_score",
        "euro_form_vs_cup_gap", "euro_weighted_attack", "euro_weighted_defense",
    ]
    for p in prefixes:
        for ef in euro_features:
            cols.append(f"{p}{ef}")

    return cols


# ══════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════
def _avg(values: list) -> float:
    clean = [_safe(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else 0.0
