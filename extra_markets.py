"""Moteur complémentaire historique SIMBET.

Ce module reprend la logique existante de app.py sans importer Streamlit, pour que
l'API FastAPI puisse enrichir les analyses J/J+1 avec les mêmes marchés
complémentaires que l'ancienne application.
"""

import logging
import math

log = logging.getLogger("SmartSim.extra_markets")

# ══════════════════════════════════════════════
# AVANTAGE DOMICILE PAR LIGUE (calibré depuis l'historique)
# ══════════════════════════════════════════════
_LEAGUE_HOME_ADV_CACHE = None
_DEFAULT_HOME_ADV = 1.10  # Multiplicateur par défaut sur lambda_home

def _compute_league_home_advantage() -> dict:
    """
    Scanne l'historique pour calculer le ratio (buts_home / buts_away) par ligue.
    Retourne un multiplicateur de lambda_home par league_id.
    Borné [1.03, 1.30] pour éviter les valeurs aberrantes.
    """
    global _LEAGUE_HOME_ADV_CACHE
    if _LEAGUE_HOME_ADV_CACHE is not None:
        return _LEAGUE_HOME_ADV_CACHE

    advantages = {}
    try:
        from supabase_db import _load_local_cache
        history = _load_local_cache()
    except Exception:
        history = {}

    # Agréger buts home/away par ligue
    by_league = {}
    for entry in history.values():
        lid = entry.get("league_id")
        result = entry.get("result", {}) or {}
        hg, ag = result.get("home_goals"), result.get("away_goals")
        if lid is None or hg is None or ag is None:
            continue
        bucket = by_league.setdefault(lid, {"home": 0, "away": 0, "n": 0})
        bucket["home"] += int(hg)
        bucket["away"] += int(ag)
        bucket["n"] += 1

    for lid, b in by_league.items():
        if b["n"] < 30 or b["away"] <= 0:
            continue  # Échantillon insuffisant
        # Ratio buts home / buts away — capture l'avantage domicile structurel
        ratio = b["home"] / b["away"]
        advantages[lid] = max(1.03, min(1.30, ratio))

    _LEAGUE_HOME_ADV_CACHE = advantages
    log.info("Avantage domicile calibré par ligue : %d ligues", len(advantages))
    return advantages


def _get_home_advantage(league_id) -> float:
    """Récupère le multiplicateur d'avantage domicile pour une ligue."""
    if league_id is None:
        return _DEFAULT_HOME_ADV
    try:
        return _compute_league_home_advantage().get(int(league_id), _DEFAULT_HOME_ADV)
    except (ValueError, TypeError):
        return _DEFAULT_HOME_ADV


# ══════════════════════════════════════════════
# MOTEUR D'INTELLIGENCE — NOUVEAUX MARCHÉS
# ══════════════════════════════════════════════
def compute_extra_markets(features: dict, proba_o25: float, league_id=None) -> dict:
    """
    Calcule Over 1.5, BTTS, pronostic vainqueur et SECURE BET
    à partir des features et de la proba Over 2.5 (modèle ML).
    Chaque marché utilise un moteur multi-facteurs dédié.
    """
    # ── Extraction de toutes les features utiles ──
    home_xg = features.get("home_avg_xg", 0) or 0
    away_xg = features.get("away_avg_xg", 0) or 0
    home_scored = features.get("home_avg_scored", 0) or 0
    away_scored = features.get("away_avg_scored", 0) or 0
    home_conceded = features.get("home_avg_conceded", 0) or 0
    away_conceded = features.get("away_avg_conceded", 0) or 0
    btts_rate = features.get("combined_btts_rate", 0) or 0
    o25_rate = features.get("combined_over25_rate", 0) or 0
    combined_xg = home_xg + away_xg

    home_weighted_scored = features.get("home_weighted_avg_scored", 0) or home_scored
    away_weighted_scored = features.get("away_weighted_avg_scored", 0) or away_scored
    home_weighted_conceded = features.get("home_weighted_avg_conceded", 0) or home_conceded
    away_weighted_conceded = features.get("away_weighted_avg_conceded", 0) or away_conceded
    home_btts = features.get("home_btts_rate", 0) or 0
    away_btts = features.get("away_btts_rate", 0) or 0
    home_o25 = features.get("home_over25_rate", 0) or 0
    away_o25 = features.get("away_over25_rate", 0) or 0
    home_cs = features.get("home_clean_sheet_rate", 0) or 0
    away_cs = features.get("away_clean_sheet_rate", 0) or 0
    home_avg_total = features.get("home_avg_total_goals", 0) or 0
    away_avg_total = features.get("away_avg_total_goals", 0) or 0
    home_conv = features.get("home_conversion_rate", 0) or 0
    away_conv = features.get("away_conversion_rate", 0) or 0
    home_shots_on = features.get("home_avg_shots_on_goal", 0) or 0
    away_shots_on = features.get("away_avg_shots_on_goal", 0) or 0
    home_shots_box = features.get("home_avg_shots_inside_box", 0) or 0
    away_shots_box = features.get("away_avg_shots_inside_box", 0) or 0
    home_poss = features.get("home_avg_possession", 50) or 50
    away_poss = features.get("away_avg_possession", 50) or 50
    home_chaos = features.get("home_chaos_factor", 0) or 0
    away_chaos = features.get("away_chaos_factor", 0) or 0
    combined_chaos = features.get("combined_chaos", 0) or 0
    match_openness = features.get("match_openness", 0) or 0
    defense_weakness = features.get("defense_weakness", 0) or 0
    home_def_press = features.get("home_defensive_pressure", 0) or 0
    away_def_press = features.get("away_defensive_pressure", 0) or 0
    home_goal_vol = features.get("home_goal_volatility", 0) or 0
    away_goal_vol = features.get("away_goal_volatility", 0) or 0
    home_g_last15 = features.get("home_avg_goals_last_15", 0) or 0
    away_g_last15 = features.get("away_avg_goals_last_15", 0) or 0
    combined_g_last15 = features.get("combined_goals_last_15", 0) or 0
    h2h_avg_goals = features.get("h2h_avg_total_goals", 0) or 0
    h2h_over25 = features.get("h2h_over25_rate", 0) or 0
    h2h_btts = features.get("h2h_btts_rate", 0) or 0
    h2h_count = features.get("h2h_count", 0) or 0
    home_fatigue = features.get("home_fatigue_score", 0) or 0
    away_fatigue = features.get("away_fatigue_score", 0) or 0
    rest_diff = features.get("rest_diff", 0) or 0
    ref_avg_goals = features.get("ref_avg_total_goals", 0) or 0
    ref_o25 = features.get("ref_over25_rate", 0) or 0
    ref_has_data = features.get("ref_has_data", 0) or 0
    home_consistency = features.get("home_scoring_consistency", 0) or 0
    away_consistency = features.get("away_scoring_consistency", 0) or 0
    home_danger_poss = features.get("home_dangerous_possession_idx", 0) or 0
    away_danger_poss = features.get("away_dangerous_possession_idx", 0) or 0
    odds_implied = features.get("odds_implied_prob_over", 0) or 0

    # ── Plafond global calibré : 92% par défaut, mais si on a une cote bookmaker
    # fiable, on utilise sa proba implicite + 5% comme borne haute (le marché sait).
    _MAX_PROBA = 0.92
    if features.get("odds_has_data") and odds_implied > 0:
        # Le marché ne va presque jamais au-delà de implied + 5% sur le réel
        _MAX_PROBA = min(0.95, max(0.75, odds_implied + 0.05))

    # ── Over 2.5 plafonne ──
    proba_o25 = min(_MAX_PROBA, proba_o25)

    # ── Pénalité dead rubber : matchs sans enjeu en fin de saison
    # = baisse l'attente de buts (-0.5 but/match historiquement, Goddard 2005)
    home_dead = features.get("home_dead_rubber", 0)
    away_dead = features.get("away_dead_rubber", 0)
    dead_rubber_factor = 1.0
    if home_dead and away_dead:
        dead_rubber_factor = 0.85  # Les deux équipes sans enjeu
    elif home_dead or away_dead:
        dead_rubber_factor = 0.93  # Une seule équipe sans enjeu

    # ══════════════════════════════════════════════════════
    # OVER 1.5 — Moteur multi-facteurs (9 signaux)
    # ══════════════════════════════════════════════════════
    # Signal 1 : Poisson — P(total >= 2) avec lambda enrichi
    lam_total = max(0.5, (
        0.30 * combined_xg +
        0.25 * (home_weighted_scored + away_weighted_scored) +
        0.25 * (home_conceded + away_conceded) / 2 +
        0.20 * (home_avg_total + away_avg_total) / 2
    ) * dead_rubber_factor)
    p0 = math.exp(-lam_total)
    p1 = lam_total * math.exp(-lam_total)
    sig_poisson_o15 = max(0.05, 1 - (p0 + p1))

    # Signal 2 : Historique Over 2.5 combiné (si Over 2.5 frequent → Over 1.5 quasi certain)
    sig_o25_hist = min(0.98, o25_rate + 0.20)

    # Signal 3 : Moyenne de buts par match des deux equipes
    avg_total_combined = (home_avg_total + away_avg_total) / 2
    sig_avg_goals = min(0.98, 0.40 + avg_total_combined * 0.15)

    # Signal 4 : Tirs cadrés combinés (plus de tirs = plus de chances de buts)
    total_shots_on = home_shots_on + away_shots_on
    sig_shots = min(0.95, 0.35 + total_shots_on * 0.04)

    # Signal 5 : Faiblesse défensive (deux mauvaises défenses = buts)
    sig_def_weak = min(0.95, 0.40 + defense_weakness * 0.30)

    # Signal 6 : Buts en fin de match (goals_last_15 = buts après 75e minute)
    sig_late_goals = min(0.92, 0.45 + combined_g_last15 * 0.20)

    # Signal 7 : H2H historique (si confrontations à buts)
    sig_h2h_o15 = 0.65  # Neutre par defaut
    if h2h_count >= 2:
        sig_h2h_o15 = min(0.95, 0.40 + h2h_avg_goals * 0.12)

    # Signal 8 : Arbitre permissif
    sig_ref = 0.65  # Neutre
    if ref_has_data:
        sig_ref = min(0.95, 0.35 + ref_avg_goals * 0.12)

    # Signal 9 : Match ouvert (les deux equipes attaquent → buts)
    sig_open = min(0.95, 0.40 + match_openness * 0.25)

    # Pondération finale Over 1.5
    w_h2h = 0.08 if h2h_count >= 2 else 0.0
    w_ref = 0.06 if ref_has_data else 0.0
    w_base = 1.0 - w_h2h - w_ref
    proba_o15 = (
        w_base * (
            0.25 * sig_poisson_o15 +
            0.15 * sig_o25_hist +
            0.15 * sig_avg_goals +
            0.12 * sig_shots +
            0.12 * sig_def_weak +
            0.08 * sig_late_goals +
            0.13 * sig_open
        ) +
        w_h2h * sig_h2h_o15 +
        w_ref * sig_ref
    )

    # Borne basse mathématique : O1.5 ≥ O2.5 + P(N=2) (Poisson exact)
    # P(N=2) = λ² · e^(-λ) / 2
    p2_poisson = (lam_total ** 2) * math.exp(-lam_total) / 2.0
    proba_o15 = max(proba_o15, proba_o25 + p2_poisson)

    # Blend avec la cote bookmaker O1.5 (signal très puissant si dispo)
    odds_o15_p = features.get("odds_implied_prob_o15", 0) or 0
    odds_o15_has = features.get("odds_o15_has_data", 0)
    if odds_o15_has and odds_o15_p > 0:
        proba_o15 = 0.70 * proba_o15 + 0.30 * odds_o15_p

    # Blend ML O1.5 si modèle dispo (priorité au ML calibré)
    ml_o15 = features.get("ml_proba_o15")
    if ml_o15 is not None and ml_o15 > 0:
        proba_o15 = 0.65 * ml_o15 + 0.35 * proba_o15

    proba_o15 = min(_MAX_PROBA, max(0.10, proba_o15))

    # ══════════════════════════════════════════════════════
    # BTTS (Both Teams To Score) — Moteur multi-facteurs (10 signaux)
    # ══════════════════════════════════════════════════════
    # Signal 1 : BTTS historique combiné (le plus fiable)
    sig_btts_hist = btts_rate

    # Signal 2 : BTTS individuel par equipe
    sig_btts_indiv = (home_btts + away_btts) / 2

    # Signal 3 : Force offensive croisée vs défense adverse
    # Si HOME marque beaucoup ET AWAY défend mal → HOME marque
    # Si AWAY marque beaucoup ET HOME défend mal → AWAY marque
    home_attack_vs_def = min(1.0, home_weighted_scored * (1 - away_cs) * 0.8)
    away_attack_vs_def = min(1.0, away_weighted_scored * (1 - home_cs) * 0.8)
    # BTTS = les DEUX marquent → produit des probabilites individuelles
    p_home_scores = min(0.98, 0.30 + home_attack_vs_def * 0.50 + home_xg * 0.15)
    p_away_scores = min(0.98, 0.30 + away_attack_vs_def * 0.50 + away_xg * 0.15)
    sig_cross_attack = p_home_scores * p_away_scores

    # Signal 4 : Anti clean-sheet — si les deux equipes encaissent regulierement
    sig_anti_cs = (1 - home_cs) * (1 - away_cs)

    # Signal 5 : xG individuels — si chaque equipe a un xG decent
    p_home_xg_scores = min(0.96, 1 - math.exp(-max(home_xg, 0.1)))
    p_away_xg_scores = min(0.96, 1 - math.exp(-max(away_xg, 0.1)))
    sig_xg_btts = p_home_xg_scores * p_away_xg_scores

    # Signal 6 : Tirs cadrés par equipe (chacune doit tirer pour marquer)
    p_home_shots = min(0.95, 0.20 + home_shots_on * 0.10)
    p_away_shots = min(0.95, 0.20 + away_shots_on * 0.10)
    sig_shots_btts = p_home_shots * p_away_shots

    # Signal 7 : Chaos factor (matchs chaotiques = buts des deux cotés)
    sig_chaos_btts = min(0.90, 0.30 + combined_chaos * 0.35)

    # Signal 8 : H2H BTTS historique
    sig_h2h_btts = 0.45  # Neutre
    if h2h_count >= 2:
        sig_h2h_btts = h2h_btts

    # Signal 9 : Efficacité offensive (conversion rate élevée des deux equipes)
    home_eff = min(0.90, 0.30 + home_conv * 2.0 + home_shots_box * 0.03)
    away_eff = min(0.90, 0.30 + away_conv * 2.0 + away_shots_box * 0.03)
    sig_efficiency = home_eff * away_eff

    # Signal 10 : Buts en fin de match des deux cotés
    sig_late_btts = min(0.85, 0.30 + (home_g_last15 + away_g_last15) * 0.20)

    # Pondération finale BTTS
    w_h2h_b = 0.08 if h2h_count >= 2 else 0.0
    w_base_b = 1.0 - w_h2h_b
    proba_btts = (
        w_base_b * (
            0.18 * sig_btts_hist +
            0.10 * sig_btts_indiv +
            0.18 * sig_cross_attack +
            0.12 * sig_anti_cs +
            0.14 * sig_xg_btts +
            0.08 * sig_shots_btts +
            0.06 * sig_chaos_btts +
            0.06 * sig_efficiency +
            0.08 * sig_late_btts
        ) +
        w_h2h_b * sig_h2h_btts
    )

    # Blend avec la cote bookmaker BTTS si disponible
    odds_btts_p = features.get("odds_implied_prob_btts", 0) or 0
    odds_btts_has = features.get("odds_btts_has_data", 0)
    if odds_btts_has and odds_btts_p > 0:
        proba_btts = 0.70 * proba_btts + 0.30 * odds_btts_p

    # Blend ML BTTS si modèle dispo
    ml_btts = features.get("ml_proba_btts")
    if ml_btts is not None and ml_btts > 0:
        proba_btts = 0.65 * ml_btts + 0.35 * proba_btts

    proba_btts = min(_MAX_PROBA, max(0.05, proba_btts))

    # ══════════════════════════════════════════════════════
    # PRONOSTIC VAINQUEUR AVANCÉ — Multi-facteurs pondérés
    # ══════════════════════════════════════════════════════
    # 1) Base Poisson avec lambda enrichi (xG + buts + défense adverse)
    home_weighted_scored = features.get("home_weighted_avg_scored", 0) or home_scored
    away_weighted_scored = features.get("away_weighted_avg_scored", 0) or away_scored
    home_weighted_conceded = features.get("home_weighted_avg_conceded", 0) or home_conceded
    away_weighted_conceded = features.get("away_weighted_avg_conceded", 0) or away_conceded

    # Forme ultra-récente (3 derniers matchs) — signal puissant à court terme
    home_form3 = features.get("home_form_last3_scored", 0) or home_weighted_scored
    away_form3 = features.get("away_form_last3_scored", 0) or away_weighted_scored

    # Splits contextuels : home dans ses matchs à domicile, away dans ses matchs à l'extérieur
    # Si l'échantillon contextuel est trop faible (<3), on retombe sur les stats globales
    home_ctx_n = features.get("home_ctx_sample_size", 0) or 0
    away_ctx_n = features.get("away_ctx_sample_size", 0) or 0
    home_ctx_scored = features.get("home_ctx_avg_scored", 0) or home_weighted_scored
    home_ctx_conceded = features.get("home_ctx_avg_conceded", 0) or home_weighted_conceded
    away_ctx_scored = features.get("away_ctx_avg_scored", 0) or away_weighted_scored
    away_ctx_conceded = features.get("away_ctx_avg_conceded", 0) or away_weighted_conceded
    # Mix contextuel/global : poids contextuel = min(0.6, n/10)
    w_home_ctx = min(0.6, home_ctx_n / 10.0) if home_ctx_n >= 3 else 0.0
    w_away_ctx = min(0.6, away_ctx_n / 10.0) if away_ctx_n >= 3 else 0.0
    home_eff_scored = w_home_ctx * home_ctx_scored + (1 - w_home_ctx) * home_weighted_scored
    home_eff_conceded = w_home_ctx * home_ctx_conceded + (1 - w_home_ctx) * home_weighted_conceded
    away_eff_scored = w_away_ctx * away_ctx_scored + (1 - w_away_ctx) * away_weighted_scored
    away_eff_conceded = w_away_ctx * away_ctx_conceded + (1 - w_away_ctx) * away_weighted_conceded

    # Absents : pondération avancée si dispo (titulaires manquants pèsent plus)
    home_missing_impact = (
        features.get("home_missing_weighted_impact")
        or features.get("home_missing_impact", 0) or 0
    )
    away_missing_impact = (
        features.get("away_missing_weighted_impact")
        or features.get("away_missing_impact", 0) or 0
    )

    # Bonus/malus formation : offensive +5%, défensive -5%
    home_form_offensive = features.get("home_formation_offensive", 0) or 0
    away_form_offensive = features.get("away_formation_offensive", 0) or 0
    home_form_factor = 1.0 + 0.05 * home_form_offensive
    away_form_factor = 1.0 + 0.05 * away_form_offensive

    lam_home = max(0.25, (
        0.35 * home_xg +
        0.25 * home_eff_scored +
        0.20 * away_eff_conceded +
        0.20 * home_form3
    ) * (1.0 - 0.15 * home_missing_impact) * dead_rubber_factor * home_form_factor)

    lam_away = max(0.25, (
        0.35 * away_xg +
        0.25 * away_eff_scored +
        0.20 * home_eff_conceded +
        0.20 * away_form3
    ) * (1.0 - 0.15 * away_missing_impact) * dead_rubber_factor * away_form_factor)

    # 2) Avantage domicile — calibré par ligue depuis l'historique + bonus possession
    home_poss = features.get("home_avg_possession", 50) or 50
    away_poss = features.get("away_avg_possession", 50) or 50
    league_adv = _get_home_advantage(league_id)
    poss_bonus = 1.0 + 0.02 * max(0, (home_poss - 50) / 10)  # 0-2% selon possession
    lam_home *= league_adv * poss_bonus

    # 2bis) Strength of Schedule — pondération par force défensive de l'adversaire
    # Utilise les buts encaissés par match (gapg) du classement
    # Référence ligue ≈ 1.3 buts encaissés/match
    if features.get("standings_has_data", 0):
        away_def_strength = features.get("away_gapg_at_away", 0) or 0
        home_def_strength = features.get("home_gapg_at_home", 0) or 0
        if away_def_strength > 0:
            # Adversaire faible défensivement → bonus lambda_home
            sos_home = max(0.85, min(1.20, away_def_strength / 1.3))
            lam_home *= sos_home
        if home_def_strength > 0:
            sos_away = max(0.85, min(1.20, home_def_strength / 1.3))
            lam_away *= sos_away

        # 2ter) Différence de niveau (rang) — petite amplification
        # rank_diff < 0 → home mieux classé → bonus
        rank_diff = features.get("rank_diff", 0) or 0
        if abs(rank_diff) >= 5:
            adj = min(0.10, abs(rank_diff) * 0.005)  # max ±10%
            if rank_diff < 0:  # home mieux classé
                lam_home *= 1.0 + adj
                lam_away *= 1.0 - adj * 0.5
            else:
                lam_away *= 1.0 + adj
                lam_home *= 1.0 - adj * 0.5

    # 3) Ajustement fatigue / repos
    home_fatigue = features.get("home_fatigue_score", 0) or 0
    away_fatigue = features.get("away_fatigue_score", 0) or 0
    rest_diff = features.get("rest_diff", 0) or 0
    if rest_diff > 1:
        lam_home *= 1.03  # Avantage repos domicile
        lam_away *= 0.97
    elif rest_diff < -1:
        lam_home *= 0.97
        lam_away *= 1.03
    if home_fatigue > 0.6:
        lam_home *= 0.95
    if away_fatigue > 0.6:
        lam_away *= 0.95

    # 4) Force défensive (clean sheet rate élevé = moins de buts encaissés)
    home_cs = features.get("home_clean_sheet_rate", 0) or 0
    away_cs = features.get("away_clean_sheet_rate", 0) or 0
    home_def_press = features.get("home_defensive_pressure", 0) or 0
    away_def_press = features.get("away_defensive_pressure", 0) or 0
    if away_cs > 0.4:
        lam_home *= max(0.80, 1.0 - away_cs * 0.25)
    if home_cs > 0.4:
        lam_away *= max(0.80, 1.0 - home_cs * 0.25)

    # 5) Efficacité offensive (conversion rate + tirs cadrés)
    home_conv = features.get("home_conversion_rate", 0) or 0
    away_conv = features.get("away_conversion_rate", 0) or 0
    home_shots_on = features.get("home_avg_shots_on_goal", 0) or 0
    away_shots_on = features.get("away_avg_shots_on_goal", 0) or 0
    if home_conv > 0.15 and home_shots_on > 4:
        lam_home *= 1.05
    if away_conv > 0.15 and away_shots_on > 4:
        lam_away *= 1.05

    # 6) Grille Poisson (0-8 buts par equipe) avec correction Dixon-Coles
    # τ(h, a, λh, λa, ρ) corrige la sous-estimation des faibles scores corrélés
    exp_h = math.exp(-lam_home)
    exp_a = math.exp(-lam_away)
    ph = [exp_h * (lam_home ** k) / math.factorial(k) for k in range(9)]
    pa = [exp_a * (lam_away ** k) / math.factorial(k) for k in range(9)]
    # ρ standard = -0.10 (valeur typique de la littérature foot)
    rho = -0.10
    def _dc_tau(h, a):
        if h == 0 and a == 0:
            return 1.0 - lam_home * lam_away * rho
        if h == 0 and a == 1:
            return 1.0 + lam_home * rho
        if h == 1 and a == 0:
            return 1.0 + lam_away * rho
        if h == 1 and a == 1:
            return 1.0 - rho
        return 1.0

    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    for h in range(9):
        for a in range(9):
            p_match = ph[h] * pa[a] * _dc_tau(h, a)
            if h > a:
                p_home_win += p_match
            elif h == a:
                p_draw += p_match
            else:
                p_away_win += p_match

    # 7) Ajustement H2H — utilise les taux réels calculés (plus de 0.25 hardcodé)
    h2h_count = features.get("h2h_count", 0) or 0
    h2h_home_wr = features.get("h2h_home_win_rate", 0) or 0
    h2h_away_wr = features.get("h2h_away_win_rate", 0) or 0
    h2h_draw_r = features.get("h2h_draw_rate", 0) or 0
    # Vérifier la cohérence des taux H2H (doit sommer à ~1.0)
    h2h_total = h2h_home_wr + h2h_away_wr + h2h_draw_r
    if h2h_count >= 3 and h2h_total > 0:
        # Normaliser si nécessaire
        if abs(h2h_total - 1.0) > 0.05:
            h2h_home_wr /= h2h_total
            h2h_away_wr /= h2h_total
            h2h_draw_r /= h2h_total
        h2h_weight = min(0.20, 0.04 * h2h_count)  # Max 20% d'influence
        p_home_win = (1 - h2h_weight) * p_home_win + h2h_weight * h2h_home_wr
        p_away_win = (1 - h2h_weight) * p_away_win + h2h_weight * h2h_away_wr
        p_draw = (1 - h2h_weight) * p_draw + h2h_weight * h2h_draw_r

    # 8) Ajustement régularité (scoring consistency)
    home_consistency = features.get("home_scoring_consistency", 0) or 0
    away_consistency = features.get("away_scoring_consistency", 0) or 0
    if home_consistency > 0.7 and away_consistency < 0.4:
        p_home_win *= 1.05
    elif away_consistency > 0.7 and home_consistency < 0.4:
        p_away_win *= 1.05

    # 9) Blend ML 1X2 si modèle dispo (60% ML + 40% Poisson selon recommandation Constantinou)
    ml_h = features.get("ml_winner_home")
    ml_d = features.get("ml_winner_draw")
    ml_a = features.get("ml_winner_away")
    if ml_h is not None and ml_d is not None and ml_a is not None:
        # Normaliser le Poisson d'abord pour blender proprement
        total_p_pois = p_home_win + p_draw + p_away_win
        if total_p_pois > 0:
            p_home_win /= total_p_pois
            p_draw /= total_p_pois
            p_away_win /= total_p_pois
        p_home_win = 0.60 * ml_h + 0.40 * p_home_win
        p_draw = 0.60 * ml_d + 0.40 * p_draw
        p_away_win = 0.60 * ml_a + 0.40 * p_away_win

    # 10) Normalisation finale
    total_p = p_home_win + p_draw + p_away_win
    if total_p > 0:
        p_home_win /= total_p
        p_draw /= total_p
        p_away_win /= total_p
    # Plafonner APRÈS normalisation (sans renormaliser — ça ne ferait que déplacer le problème)
    p_home_win = min(_MAX_PROBA, p_home_win)
    p_draw = min(_MAX_PROBA, p_draw)
    p_away_win = min(_MAX_PROBA, p_away_win)

    # 11) Déterminer le vainqueur probable
    if p_home_win >= p_away_win and p_home_win >= p_draw:
        winner = "home"
        winner_proba = p_home_win
    elif p_away_win >= p_home_win and p_away_win >= p_draw:
        winner = "away"
        winner_proba = p_away_win
    else:
        winner = "draw"
        winner_proba = p_draw

    return {
        "proba_o15": round(proba_o15, 4),
        "proba_btts": round(proba_btts, 4),
        "p_home_win": round(p_home_win, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away_win, 4),
        "winner": winner,
        "winner_proba": round(winner_proba, 4),
        "is_secure_bet": False,  # Supprime — garde la cle pour compatibilite
        "combined_xg": round(combined_xg, 2),
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
    }
