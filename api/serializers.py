"""
Serializers JSON — convertit les dicts internes (sortie de model.predict_today
ou load_bet_history) en payloads propres pour le frontend.

2 niveaux :
- serialize_match_summary  → liste compacte (page Tous les matchs / Smart Sim)
- serialize_match_detail   → fiche match complète (page /match/[id])
"""

import logging
from typing import Optional

log = logging.getLogger("SmartSim.serializers")


# Phase 2 — Branchement parallèle du modèle Over 1.5 candidate (H_clip_G_70).
# Chargement paresseux + fallback graceful si le .pkl ou le module est absent.
def _try_predict_over15_candidate(features, proba_o25_raw, poisson_o15) -> Optional[float]:
    """
    Retourne la probabilité Over 1.5 du candidate (max(blend, p_o25_raw))
    ou None si :
      - le module inference n'est pas disponible
      - le .pkl n'existe pas (déploiement HF partiel)
      - une exception survient à l'inférence
    """
    try:
        from inference.over15_candidate import predict_over15_candidate
        result = predict_over15_candidate(
            features=features or {},
            proba_o25_raw=proba_o25_raw or 0.0,
            poisson_o15=poisson_o15,
        )
        return result.get("over15_candidate")
    except Exception as e:
        # Log une seule fois par process (DEBUG niveau, pas WARNING) pour éviter le bruit
        log.debug("over15_candidate prediction skipped: %s", e)
        return None


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _optional_float(v):
    try:
        if v is None or v == "":
            return None
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _safe_int(v, default=None):
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _confidence_label(value):
    value = _safe_float(value)
    if value >= 0.70:
        return "forte"
    if value >= 0.60:
        return "moyenne"
    return "faible"


def display_confidence_label(value):
    value = _safe_float(value)
    if value >= 0.75:
        return "Confiance très forte"
    if value >= 0.65:
        return "Confiance forte"
    if value >= 0.55:
        return "Confiance correcte"
    return "Confiance modérée"


def calibrate_display_probability(market: str, raw_probability):
    """
    Calibre uniquement la probabilité affichée au public.
    La probabilité brute reste intacte pour le modèle, le tri et les backtests.
    """
    raw = _optional_float(raw_probability)
    if raw is None:
        return None

    if market != "over_25":
        return raw

    calibration = [
        (0.50, 0.60, 0.49),
        (0.60, 0.70, 0.51),
        (0.70, 0.80, 0.66),
        (0.80, 0.90, 0.81),
        (0.90, 1.01, 0.67),
    ]
    for low, high, display in calibration:
        if low <= raw < high:
            return round(min(max(display, 0.0), 0.85), 4)

    return round(min(max(raw, 0.0), 0.85), 4)


def build_l2m_selection(btts_probability) -> dict:
    probability = _optional_float(btts_probability)
    is_selection = probability is not None and probability >= 0.60
    return {
        "is_selection": is_selection,
        "pick": "L2M" if is_selection else "",
        "probability": probability,
        "confidence": "forte" if probability is not None and probability >= 0.70 else "moyenne" if is_selection else "faible",
        "label": "Lecture L2M" if is_selection else "",
        "reason": "La lecture L2M est mise en avant uniquement lorsque la probabilité atteint au moins 60%." if is_selection else "",
    }


def detect_result_selection_from_probabilities(home_win, draw, away_win) -> dict:
    """
    Sélectionne uniquement les lectures résultat suffisamment nettes.
    Utilise les probabilités déjà calculées par l'ancien moteur complémentaire.
    """
    probs = {
        "1": _optional_float(home_win),
        "N": _optional_float(draw),
        "2": _optional_float(away_win),
    }
    if any(value is None for value in probs.values()):
        return {
            "type": None,
            "pick": "",
            "label": "",
            "probability": None,
            "confidence": "faible",
            "is_result_selection": False,
        }

    labels = {
        "1": "Victoire domicile",
        "N": "Match nul",
        "2": "Victoire extérieur",
        "1N": "Victoire ou nul",
        "N2": "Nul ou victoire extérieur",
        "12": "Domicile ou extérieur",
    }

    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    best_pick, best_value = ranked[0]
    second_value = ranked[1][1]
    margin = best_value - second_value

    if best_pick == "N":
        single_ok = best_value >= 0.46 and margin >= 0.10
    else:
        single_ok = best_value >= 0.58 and margin >= 0.12

    if single_ok:
        return {
            "type": "single",
            "pick": best_pick,
            "label": labels[best_pick],
            "probability": round(best_value, 4),
            "confidence": _confidence_label(best_value),
            "is_result_selection": True,
        }

    double_candidates = [
        ("1N", probs["1"] + probs["N"], probs["2"], abs(probs["1"] - probs["N"])),
        ("N2", probs["N"] + probs["2"], probs["1"], abs(probs["N"] - probs["2"])),
        ("12", probs["1"] + probs["2"], probs["N"], abs(probs["1"] - probs["2"])),
    ]
    double_candidates.sort(key=lambda item: item[1], reverse=True)

    for pick, probability, excluded, spread in double_candidates:
        if probability >= 0.74 and excluded <= 0.26 and spread <= 0.14 and max(probs.values()) >= 0.42:
            return {
                "type": "double_chance",
                "pick": pick,
                "label": labels[pick],
                "probability": round(probability, 4),
                "confidence": _confidence_label(probability),
                "is_result_selection": True,
            }

    return {
        "type": None,
        "pick": "",
        "label": "",
        "probability": None,
        "confidence": "faible",
        "is_result_selection": False,
    }


# ══════════════════════════════════════════════
# RÉSUMÉ (pour les listes)
# ══════════════════════════════════════════════
def serialize_match_summary(m: dict) -> dict:
    """Format compact : ce qu'il faut pour afficher une carte dans une liste."""
    pred = m.get("prediction", {}) or {}
    smart = pred.get("smart_bet", {}) or {}
    extra = m.get("_extra") or {}
    home = m.get("home_team", {}) or {}
    away = m.get("away_team", {}) or {}
    odds = m.get("odds_data") or {}
    over25_raw = _optional_float(pred.get("proba_over25"))
    over25_display = calibrate_display_probability("over_25", over25_raw)
    btts_probability = _optional_float(extra.get("proba_btts"))

    # Phase 2 — Over 1.5 candidate (parallel ML model, fallback-safe)
    over15_legacy = _optional_float(extra.get("proba_o15"))
    over15_candidate = _try_predict_over15_candidate(
        features=m.get("features"),
        proba_o25_raw=pred.get("proba_over25"),
        poisson_o15=extra.get("proba_o15"),
    )
    # Préférence : candidate si dispo, sinon legacy
    over15_display = over15_candidate if over15_candidate is not None else over15_legacy

    return {
        "fixture_id": str(m.get("fixture_id", "")),
        "league": {
            "id": m.get("league_id"),
            "name": m.get("league_name", ""),
            "flag": m.get("league_flag", ""),
            "country": m.get("league_country", ""),
        },
        "date": m.get("date", ""),
        "venue": m.get("venue", ""),
        "status": {
            "code": m.get("match_status", "NS"),
            "elapsed": m.get("match_elapsed"),
        },
        "score": {
            "home": _safe_int(m.get("current_home_goals")),
            "away": _safe_int(m.get("current_away_goals")),
        },
        "home_team": {
            "id": home.get("id"),
            "name": home.get("name", ""),
            "logo": home.get("logo", ""),
        },
        "away_team": {
            "id": away.get("id"),
            "name": away.get("name", ""),
            "logo": away.get("logo", ""),
        },
        "probabilities": {
            "over_25": round(_safe_float(pred.get("proba_over25")), 4),
            "over_25_raw": over25_raw,
            "over_25_display": over25_display,
            "over_25_confidence_label": display_confidence_label(over25_display),
            "over_15": over15_display,             # candidate si dispo, sinon legacy
            "over_15_legacy": over15_legacy,       # toujours la valeur historique
            "over_15_candidate": over15_candidate, # null si modèle absent
            "over_15_source": "candidate" if over15_candidate is not None else "legacy",
            "btts":    btts_probability,
            "home_win": _optional_float(extra.get("p_home_win")),
            "draw":     _optional_float(extra.get("p_draw")),
            "away_win": _optional_float(extra.get("p_away_win")),
        },
        "predicted_winner": extra.get("winner") or "",
        "winner_proba": _optional_float(extra.get("winner_proba")),
        "result_selection": detect_result_selection_from_probabilities(
            extra.get("p_home_win"),
            extra.get("p_draw"),
            extra.get("p_away_win"),
        ),
        "l2m_selection": build_l2m_selection(btts_probability),
        "is_smart_bet": bool(smart.get("is_smart_bet", False)),
        "smart_bet": {
            "is_smart_bet": bool(smart.get("is_smart_bet", False)),
            "is_value": bool(smart.get("is_value", False)),
            "reason": smart.get("reason") or "",
        },
        "label": pred.get("label", ""),
        "odds": {
            "over_25": odds.get("avg_over_25"),
            "over_15": m.get("odd_over15"),
            "btts":    m.get("odd_btts"),
        },
    }


# ══════════════════════════════════════════════
# DÉTAIL (pour la fiche match /match/[id])
# ══════════════════════════════════════════════
def _form_pills(matches: list, team_id: int, n: int = 5) -> list:
    """Retourne ['W','D','L','W','W'] pour les n derniers matchs."""
    pills = []
    for past in (matches or [])[:n]:
        try:
            mh_id = past.get("teams", {}).get("home", {}).get("id")
            hg = past.get("goals", {}).get("home")
            ag = past.get("goals", {}).get("away")
            if hg is None or ag is None:
                continue
            scored, conceded = (int(hg), int(ag)) if team_id == mh_id else (int(ag), int(hg))
            if scored > conceded:
                pills.append("W")
            elif scored == conceded:
                pills.append("D")
            else:
                pills.append("L")
        except Exception:
            continue
    return pills


def _h2h_summary(h2h: list, home_id: int, n: int = 5) -> list:
    """5 derniers face-à-face : [{date, home, away, score, winner_id}]."""
    out = []
    for past in (h2h or [])[:n]:
        try:
            ph = past.get("teams", {}).get("home", {}) or {}
            pa = past.get("teams", {}).get("away", {}) or {}
            hg = past.get("goals", {}).get("home")
            ag = past.get("goals", {}).get("away")
            if hg is None or ag is None:
                continue
            hg_i, ag_i = int(hg), int(ag)
            if hg_i > ag_i:
                winner_id = ph.get("id")
            elif ag_i > hg_i:
                winner_id = pa.get("id")
            else:
                winner_id = None
            out.append({
                "date": past.get("fixture", {}).get("date", "")[:10],
                "home": {"id": ph.get("id"), "name": ph.get("name", ""), "logo": ph.get("logo", "")},
                "away": {"id": pa.get("id"), "name": pa.get("name", ""), "logo": pa.get("logo", "")},
                "score": {"home": hg_i, "away": ag_i},
                "winner_id": winner_id,
            })
        except Exception:
            continue
    return out


def serialize_match_detail(m: dict) -> dict:
    """Format détaillé : page /match/[id]."""
    base = serialize_match_summary(m)
    pred = m.get("prediction", {}) or {}
    smart = pred.get("smart_bet", {}) or {}
    home = m.get("home_team", {}) or {}
    away = m.get("away_team", {}) or {}
    home_id = home.get("id")
    away_id = away.get("id")

    base["smart_bet"] = {
        "signals": smart.get("signals") or [],
        "signal_count": smart.get("signal_count"),
        "convergence_score": smart.get("convergence_score"),
    }
    base["top_drivers"] = pred.get("top_drivers") or []
    base["form"] = {
        "home": _form_pills(m.get("home_last_matches"), home_id),
        "away": _form_pills(m.get("away_last_matches"), away_id),
    }
    base["h2h"] = _h2h_summary(m.get("h2h"), home_id)
    base["analysis"] = {
        "commentary": m.get("_commentary") or "",
        "model": {
            "xgb": round(_safe_float(m.get("prediction", {}).get("xgb_proba")), 4),
            "lgb": round(_safe_float(m.get("prediction", {}).get("lgb_proba")), 4),
        },
    }

    # Phase 2 — match_insights enrichis (fallback-safe).
    # Injecte la proba over15_candidate dans `m` pour que build_insights l'utilise
    # comme préférence d'affichage du marché +1.5.
    try:
        from api.match_insights import build_insights
        m_with_candidate = dict(m)
        m_with_candidate["over_15_candidate"] = base["probabilities"].get("over_15_candidate")
        base["insights"] = build_insights(m_with_candidate)
    except Exception as e:
        log.debug("build_insights skipped: %s", e)
        base["insights"] = None

    return base
