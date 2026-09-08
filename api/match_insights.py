"""
Module Match Insights (Phase 2 — analyses enrichies).

Pure functions, aucune écriture, aucun appel API. Prend en entrée un dict
match (sortie de predict_today + compute_extra_markets) et renvoie un payload
d'analyse riche par marché avec :
  - probability + confidence_tier
  - headline + warning + explanations[]
  - value bet si odds disponibles
  - market_strength_score

Plus un bloc match_profile, best_market, risk_level, smart_summary.

Fallback-safe : si une donnée manque, renvoie un fallback propre sans
phrases inventées.

Tous les seuils sont calibrés selon les spec Phase 2 :
  Over 2.5 : <58 absent / 58-62 prudent / 63-68 moyen / >=69 fort
  Over 1.5 : <70 absent / 70-74 prudent / 75-79 moyen / >=80 fort
  BTTS     : <58 absent / 58-62 prudent / 63-67 moyen / >=68 fort
  Winner   : fort si proba >= 65% OU double_chance >= 72%
"""
from __future__ import annotations

from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────
# Confidence tiers (par marché)
# ──────────────────────────────────────────────────────────────────────
TIERS = {
    "over25": [(0.69, "fort"), (0.63, "moyen"), (0.58, "prudent"), (0.0, "absent")],
    "over15": [(0.80, "fort"), (0.75, "moyen"), (0.70, "prudent"), (0.0, "absent")],
    "btts":   [(0.68, "fort"), (0.63, "moyen"), (0.58, "prudent"), (0.0, "absent")],
}


def confidence_tier(market: str, probability: Optional[float]) -> str:
    """Retourne 'absent' / 'prudent' / 'moyen' / 'fort' selon les seuils."""
    if probability is None:
        return "absent"
    thresholds = TIERS.get(market)
    if not thresholds:
        return "absent"
    for threshold, label in thresholds:
        if probability >= threshold:
            return label
    return "absent"


def winner_confidence_tier(probability: Optional[float],
                            double_chance_proba: Optional[float] = None) -> str:
    """
    Winner : fort UNIQUEMENT si proba >= 0.65 OU double_chance >= 0.72.
    Sinon gradient classique.
    """
    if probability is None and double_chance_proba is None:
        return "absent"
    p = probability or 0.0
    dc = double_chance_proba or 0.0
    if p >= 0.65 or dc >= 0.72:
        return "fort"
    if p >= 0.61: return "moyen"
    if p >= 0.55: return "prudent"
    return "absent"


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _implied_prob(odd: Optional[float]) -> Optional[float]:
    """Probabilité implicite naïve depuis une cote bookmaker."""
    o = _safe_float(odd)
    if o is None or o <= 1.0:
        return None
    return 1.0 / o


def _compute_value(proba_model: Optional[float], odd: Optional[float]) -> Optional[dict]:
    """Calcule la value bet si les deux entrées sont valides."""
    if proba_model is None or proba_model <= 0:
        return None
    implied = _implied_prob(odd)
    if implied is None:
        return None
    edge = round(proba_model - implied, 4)
    return {
        "implied_prob": round(implied, 4),
        "model_prob":   round(float(proba_model), 4),
        "edge":         edge,
        "has_value":    edge >= 0.05,  # 5pts d'edge = seuil mini
    }


# ──────────────────────────────────────────────────────────────────────
# Headlines et phrases d'analyse (data-driven, pas de templates inventés)
# ──────────────────────────────────────────────────────────────────────
def _headline_over25(p: Optional[float], tier: str, features: dict) -> str:
    if p is None:
        return "Probabilité Over 2.5 indisponible."
    if tier == "fort":
        return f"Signal Over 2.5 solide ({round(p*100)}%)."
    if tier == "moyen":
        return f"Signal Over 2.5 correct ({round(p*100)}%), à confirmer."
    if tier == "prudent":
        return f"Signal Over 2.5 limité ({round(p*100)}%), prudence."
    return "Pas de signal Over 2.5 net."


def _headline_over15(p: Optional[float], tier: str) -> str:
    if p is None:
        return "Probabilité Over 1.5 indisponible."
    if tier == "fort":
        return f"Over 1.5 très probable ({round(p*100)}%)."
    if tier == "moyen":
        return f"Over 1.5 probable ({round(p*100)}%)."
    if tier == "prudent":
        return f"Over 1.5 plausible ({round(p*100)}%), à confirmer."
    return "Pas de signal Over 1.5 net."


def _headline_btts(p: Optional[float], tier: str) -> str:
    if p is None:
        return "Probabilité BTTS indisponible."
    if tier == "fort":
        return f"Les deux équipes devraient marquer ({round(p*100)}%)."
    if tier == "moyen":
        return f"Scénario BTTS plausible ({round(p*100)}%)."
    if tier == "prudent":
        return f"BTTS incertain ({round(p*100)}%)."
    return "Pas de signal BTTS net."


def _headline_winner(p_home: Optional[float], p_draw: Optional[float],
                      p_away: Optional[float], home_name: str, away_name: str) -> tuple[str, Optional[str]]:
    """Retourne (headline, match_type)."""
    items = [(p_home, home_name, "home"), (p_draw, "match nul", "draw"), (p_away, away_name, "away")]
    items = [(p, n, k) for p, n, k in items if p is not None]
    if not items:
        return ("Probabilités 1X2 indisponibles.", None)
    items.sort(key=lambda x: -x[0])
    top_p, top_n, top_k = items[0]
    second_p = items[1][0] if len(items) > 1 else 0.0
    gap = top_p - second_p
    if top_p >= 0.65 and gap >= 0.20:
        return (f"Favori clair : {top_n} ({round(top_p*100)}%).", "favori_clair")
    if top_p >= 0.55 and gap >= 0.12:
        return (f"Léger avantage à {top_n} ({round(top_p*100)}%).", "favori_leger")
    if (p_draw or 0) >= 0.30:
        return (f"Risque de nul élevé ({round((p_draw or 0)*100)}%).", "risque_nul")
    if gap < 0.08:
        return ("Match équilibré, aucune issue ne se détache clairement.", "match_ouvert")
    return (f"Match plutôt ouvert, légère préférence pour {top_n}.", "balanced")


def _explanations_over25(p: Optional[float], features: dict) -> list[str]:
    """Phrases data-driven uniquement si la feature existe."""
    out: list[str] = []
    if p is None:
        return out
    combined_xg = _safe_float(features.get("combined_xg"))
    if combined_xg is not None:
        if combined_xg >= 2.8:
            out.append(f"xG combiné élevé ({round(combined_xg,1)}) : profil offensif des deux côtés.")
        elif combined_xg <= 2.0:
            out.append(f"xG combiné modeste ({round(combined_xg,1)}) : profil prudent attendu.")
    h2h_o25 = _safe_float(features.get("h2h_over25_rate"))
    h2h_count = _safe_float(features.get("h2h_count"))
    if h2h_o25 is not None and h2h_count and h2h_count >= 3:
        if h2h_o25 >= 0.6:
            out.append(f"H2H récents majoritairement Over 2.5 ({round(h2h_o25*100)}%).")
        elif h2h_o25 <= 0.3:
            out.append(f"H2H plutôt fermés ({round(h2h_o25*100)}% d'Over 2.5).")
    avg_scored = _safe_float(features.get("home_avg_scored"))
    avg_conc = _safe_float(features.get("away_avg_conceded"))
    if avg_scored is not None and avg_conc is not None:
        if avg_scored >= 1.7 and avg_conc >= 1.5:
            out.append("Attaque domicile bien rodée face à une défense extérieure perméable.")
    return out


def _explanations_over15(p_over15: Optional[float], p_over25: Optional[float]) -> list[str]:
    out: list[str] = []
    if p_over15 is None:
        return out
    if p_over25 is not None and p_over15 - p_over25 >= 0.12:
        out.append("Over 1.5 plus sécurisé qu'Over 2.5 : le modèle voit des buts sans surpondérer 3+.")
    if p_over15 >= 0.85:
        out.append("Probabilité très élevée : seuil rarement raté sur ce profil de match.")
    return out


def _explanations_btts(p_btts: Optional[float], features: dict) -> list[str]:
    out: list[str] = []
    if p_btts is None:
        return out
    home_cs = _safe_float(features.get("home_clean_sheet_rate"))
    away_cs = _safe_float(features.get("away_clean_sheet_rate"))
    if home_cs is not None and away_cs is not None:
        if (home_cs + away_cs) / 2 <= 0.20:
            out.append("Aucune des deux équipes n'a une défense solide récemment.")
        elif home_cs >= 0.45 or away_cs >= 0.45:
            out.append("Une des deux équipes a un bon ratio clean sheets — BTTS contestable.")
    btts_rate = _safe_float(features.get("combined_btts_rate"))
    if btts_rate is not None:
        if btts_rate >= 0.6:
            out.append(f"Historique récent fortement BTTS ({round(btts_rate*100)}%).")
        elif btts_rate <= 0.35:
            out.append(f"Historique récent peu BTTS ({round(btts_rate*100)}%).")
    return out


def _explanations_winner(p_home, p_draw, p_away, match_type: Optional[str]) -> list[str]:
    out: list[str] = []
    items = [p for p in (p_home, p_draw, p_away) if p is not None]
    if not items:
        return out
    if match_type == "match_ouvert":
        out.append("Les 3 issues sont possibles — préférer un marché des buts si tu cherches plus de sécurité.")
    elif match_type == "risque_nul":
        out.append("La probabilité de match nul est inhabituellement élevée — éviter le 1X2 simple.")
    return out


def _market_strength_score(probability: Optional[float], tier: str) -> float:
    """Score 0-100 combinant proba et tier."""
    if probability is None:
        return 0.0
    tier_bonus = {"fort": 30, "moyen": 20, "prudent": 10, "absent": 0}.get(tier, 0)
    return round(min(100, probability * 70 + tier_bonus), 1)


# ──────────────────────────────────────────────────────────────────────
# Builder principal — appelé par les serializers
# ──────────────────────────────────────────────────────────────────────
def build_insights(match: dict) -> dict:
    """
    Construit le payload `insights` complet pour un match.

    Inputs attendus (tous optional / fallback-safe) :
      match["prediction"]["proba_over25"]
      match["_extra"]["proba_o15", "proba_btts", "p_home_win", "p_draw", "p_away_win", "winner"]
      match["features"] (dict feature vector, optional)
      match["odds_data"]["avg_over_25"], match["odd_over15"], match["odd_btts"]
      match["home_team"], match["away_team"]
      match.get("over_15_candidate") (Phase 2 : injecté par le serializer)

    Retourne un dict avec : summary, over25, over15, btts, l2m, result, context, smart_summary.
    """
    pred = match.get("prediction") or {}
    extra = match.get("_extra") or {}
    features = match.get("features") or {}
    odds_data = match.get("odds_data") or {}
    if not isinstance(odds_data, dict):
        odds_data = {}

    home_name = (match.get("home_team") or {}).get("name", "domicile")
    away_name = (match.get("away_team") or {}).get("name", "extérieur")

    p_over25 = _safe_float(pred.get("proba_over25"))
    p_over15_legacy = _safe_float(extra.get("proba_o15"))
    p_over15_candidate = _safe_float(match.get("over_15_candidate"))
    # Préférence affichage : candidate si disponible, sinon legacy
    p_over15 = p_over15_candidate if p_over15_candidate is not None else p_over15_legacy
    p_btts = _safe_float(extra.get("proba_btts"))
    p_home = _safe_float(extra.get("p_home_win"))
    p_draw = _safe_float(extra.get("p_draw"))
    p_away = _safe_float(extra.get("p_away_win"))

    # ─── Over 2.5 ───
    over25_tier = confidence_tier("over25", p_over25)
    over25_block = {
        "probability":       round(p_over25, 4) if p_over25 is not None else None,
        "confidence_tier":   over25_tier,
        "headline":          _headline_over25(p_over25, over25_tier, features),
        "warning":           None,
        "explanations":      _explanations_over25(p_over25, features),
        "value":             _compute_value(p_over25, odds_data.get("avg_over_25")),
        "market_strength_score": _market_strength_score(p_over25, over25_tier),
    }
    if p_over25 is not None and over25_tier == "absent" and p_over25 >= 0.40:
        over25_block["warning"] = "Probabilité trop basse pour un signal exploitable."

    # ─── Over 1.5 ───
    over15_tier = confidence_tier("over15", p_over15)
    over15_block = {
        "probability":       round(p_over15, 4) if p_over15 is not None else None,
        "confidence_tier":   over15_tier,
        "headline":          _headline_over15(p_over15, over15_tier),
        "warning":           None,
        "explanations":      _explanations_over15(p_over15, p_over25),
        "value":             _compute_value(p_over15, match.get("odd_over15")),
        "market_strength_score": _market_strength_score(p_over15, over15_tier),
        "source":            "candidate" if p_over15_candidate is not None else "legacy",
    }

    # ─── BTTS ───
    btts_tier = confidence_tier("btts", p_btts)
    btts_block = {
        "probability":       round(p_btts, 4) if p_btts is not None else None,
        "confidence_tier":   btts_tier,
        "headline":          _headline_btts(p_btts, btts_tier),
        "warning":           None,
        "explanations":      _explanations_btts(p_btts, features),
        "value":             _compute_value(p_btts, match.get("odd_btts")),
        "market_strength_score": _market_strength_score(p_btts, btts_tier),
    }
    # L2M : alias BTTS strict ≥ 60 %
    l2m_block = {
        "is_strong_pick": (p_btts is not None and p_btts >= 0.60),
        "probability":    btts_block["probability"],
        "headline":       _headline_btts(p_btts, btts_tier) if p_btts and p_btts >= 0.60
                          else "Lecture L2M non recommandée pour ce match.",
    }

    # ─── Winner ───
    headline_winner, match_type = _headline_winner(p_home, p_draw, p_away, home_name, away_name)
    # Choix du pick par argmax + tier strict
    candidates = [(p_home, "home", home_name), (p_draw, "draw", "match nul"), (p_away, "away", away_name)]
    candidates = [(p, k, n) for p, k, n in candidates if p is not None]
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        top_p, top_k, top_n = candidates[0]
        second_p = candidates[1][0] if len(candidates) > 1 else 0.0
        # Calcul double chance la plus forte
        dc_candidates = []
        if p_home is not None and p_draw is not None: dc_candidates.append(("1N", p_home + p_draw))
        if p_home is not None and p_away is not None: dc_candidates.append(("12", p_home + p_away))
        if p_draw is not None and p_away is not None: dc_candidates.append(("N2", p_draw + p_away))
        dc_best = max((p for _, p in dc_candidates), default=None)
        winner_tier = winner_confidence_tier(top_p, dc_best)
    else:
        top_p, top_k, top_n, second_p, winner_tier = None, None, None, 0.0, "absent"

    winner_block = {
        "pick":              top_k,
        "pick_label":        top_n,
        "probabilities":     {"home": round(p_home, 4) if p_home is not None else None,
                              "draw": round(p_draw, 4) if p_draw is not None else None,
                              "away": round(p_away, 4) if p_away is not None else None},
        "confidence_tier":   winner_tier,
        "match_type":        match_type,
        "headline":          headline_winner,
        "warning":           None,
        "explanations":      _explanations_winner(p_home, p_draw, p_away, match_type),
        "market_strength_score": _market_strength_score(top_p, winner_tier),
    }
    if top_p is not None and (top_p - second_p) < 0.08:
        winner_block["warning"] = "Écart entre issues trop faible — éviter le 1X2 simple."

    # ─── Best market : marché avec le market_strength_score le plus haut ───
    market_scores = [
        ("over_15", over15_block["market_strength_score"]),
        ("over_25", over25_block["market_strength_score"]),
        ("btts",    btts_block["market_strength_score"]),
        ("winner",  winner_block["market_strength_score"]),
    ]
    market_scores.sort(key=lambda x: -x[1])
    best_market, best_score = market_scores[0]
    if best_score < 30:
        best_market = None  # aucun marché digne d'intérêt

    # ─── Risk level global ───
    if winner_tier == "fort" or over25_tier == "fort" or over15_tier == "fort":
        risk_level = "faible"
    elif winner_tier == "moyen" or over25_tier == "moyen" or over15_tier == "moyen":
        risk_level = "moyen"
    else:
        risk_level = "élevé"

    # ─── Smart summary ───
    if best_market == "over_15" and over15_tier in ("fort", "moyen"):
        smart_summary = f"Le marché Over 1.5 est le plus solide sur ce match ({over15_block['probability']*100:.0f}%)."
    elif best_market == "over_25" and over25_tier in ("fort", "moyen"):
        smart_summary = f"Le marché Over 2.5 ressort comme meilleure lecture ({over25_block['probability']*100:.0f}%)."
    elif best_market == "winner" and winner_tier in ("fort", "moyen"):
        smart_summary = f"Le résultat 1X2 est lisible : {winner_block['pick_label']}."
    elif best_market == "btts" and btts_tier in ("fort", "moyen"):
        smart_summary = f"BTTS apparaît comme le marché le plus intéressant ({btts_block['probability']*100:.0f}%)."
    elif match_type == "match_ouvert":
        smart_summary = "Match très ouvert : aucune lecture n'est assez nette pour un signal fort."
    elif match_type == "risque_nul":
        smart_summary = "Probabilité de nul élevée — privilégier les marchés des buts ou éviter le match."
    else:
        smart_summary = "Pas de signal franc identifié sur ce match."

    # ─── Match profile ───
    if match_type:
        match_profile = match_type
    elif p_over25 is not None and p_over25 >= 0.70 and p_btts is not None and p_btts >= 0.65:
        match_profile = "profil_offensif"
    elif p_over25 is not None and p_over25 <= 0.45:
        match_profile = "profil_prudent"
    else:
        match_profile = "match_equilibre"

    return {
        "summary": {
            "best_market":   best_market,
            "risk_level":    risk_level,
            "match_profile": match_profile,
            "smart_summary": smart_summary,
        },
        "over25":  over25_block,
        "over15":  over15_block,
        "btts":    btts_block,
        "l2m":     l2m_block,
        "result":  winner_block,
    }
