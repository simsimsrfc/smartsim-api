"""
Routes /api/history/*

GET /api/history/dates          — Liste des dates disponibles dans l'historique
GET /api/history/{date}         — Matchs + résultats d'une date passée
GET /api/history/smart/{date}   — Smart Selections d'une date + bilan win/loss
"""

import logging
import re
from datetime import date as _date, datetime as _datetime, time as _time, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from data_fetcher import load_daily_cache, list_cached_dates
from supabase_db import (
    evaluate_prediction,
    load_bet_history,
    list_bet_history_dates,
)
from config import HISTORY_START_DATE, SMART_BET_THRESHOLD

from api.serializers import serialize_match_summary

log = logging.getLogger("SmartSim.api.history")
router = APIRouter()


# ──────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────
def _parse_history_start() -> _datetime:
    raw = str(HISTORY_START_DATE or "").strip()
    if not raw:
        return _datetime.min.replace(tzinfo=timezone.utc)
    try:
        if len(raw) == 10:
            return _datetime.combine(_date.fromisoformat(raw), _time.min).replace(tzinfo=timezone.utc)
        return _datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        log.warning("HISTORY_START_DATE invalide (%s), aucun ancien historique masqué", raw)
        return _datetime.min.replace(tzinfo=timezone.utc)


_HISTORY_START_AT = _parse_history_start()


def _parse_date(date_str: str) -> _date:
    try:
        return _date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(400, f"Date invalide : {date_str} (format attendu YYYY-MM-DD)")


def _is_smart(m: dict) -> bool:
    pred = m.get("prediction", {}) or {}
    smart = pred.get("smart_bet", {}) or {}
    proba = pred.get("proba_over25", 0) or 0
    return bool(smart.get("is_smart_bet")) or proba >= SMART_BET_THRESHOLD


def _check_over25_result(m: dict) -> str:
    """Retourne pending / won / lost / void pour la lecture +2,5."""
    # Priorité : colonne pré-calculée par sync-results
    r = m.get("_results") or {}
    if r.get("result_over25_won") is not None:
        return "won" if r["result_over25_won"] else "lost"
    status = m.get("match_status", "NS")
    hg = m.get("current_home_goals")
    ag = m.get("current_away_goals")
    return evaluate_prediction("over25", "+2.5", hg, ag, status)


def _check_winner_result(m: dict) -> str:
    """Évalue la sélection résultat, y compris les doubles chances."""
    r = m.get("_results") or {}
    if r.get("result_winner_won") is not None:
        return "won" if r["result_winner_won"] else "lost"
    summary = serialize_match_summary(m)
    result_selection = summary.get("result_selection") or {}
    pick = result_selection.get("pick") or _winner_to_pick(summary.get("predicted_winner"))
    status = m.get("match_status", "NS")
    hg = m.get("current_home_goals")
    ag = m.get("current_away_goals")
    return evaluate_prediction("result", pick, hg, ag, status)


def _history_status(value: str) -> str:
    return {
        "win": "won",
        "loss": "lost",
        "won": "won",
        "lost": "lost",
        "pending": "pending",
        "void": "void",
    }.get(value, "pending")


def _score_label(summary: dict) -> dict:
    score = summary.get("score") or {}
    home = score.get("home")
    away = score.get("away")
    if home is None or away is None:
        return {"score": "", "label": ""}

    total = int(home) + int(away)
    if home > away:
        result_label = "Victoire domicile"
    elif away > home:
        result_label = "Victoire extérieur"
    else:
        result_label = "Match nul"
    return {
        "score": f"{home} - {away}",
        "label": f"{total} buts" if total != 1 else "1 but",
        "result_label": result_label,
    }


def _date_in_range(value: str, start: Optional[_date], end: Optional[_date]) -> bool:
    try:
        current = _date.fromisoformat(str(value)[:10])
    except ValueError:
        return False
    if start and current < start:
        return False
    if end and current > end:
        return False
    return True


def _parse_history_datetime(value) -> Optional[_datetime]:
    if not value:
        return None
    raw = str(value).strip()
    raw = re.sub(
        r"\.(\d{1,5})(?=Z|[+-]\d{2}:?\d{2}$)",
        lambda match: "." + match.group(1).ljust(6, "0"),
        raw,
    )
    try:
        if len(raw) == 10:
            return _datetime.combine(_date.fromisoformat(raw), _time.min).replace(tzinfo=timezone.utc)
        parsed = _datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _history_reference_datetime(match: dict) -> Optional[_datetime]:
    for field in ("analysis_date", "created_at", "date"):
        parsed = _parse_history_datetime(match.get(field))
        if parsed is not None:
            return parsed
    return None


def _is_after_history_start(match: dict) -> bool:
    reference = _history_reference_datetime(match)
    if reference is None:
        return False
    start = _HISTORY_START_AT
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return reference >= start


def _load_real_history(start: Optional[_date] = None, end: Optional[_date] = None) -> list[dict]:
    """Charge uniquement l'historique réel bet_history Supabase, sans mocks UI."""
    try:
        available_dates = list_bet_history_dates() or []
    except Exception as e:
        log.warning("Erreur list_bet_history_dates : %s", e)
        available_dates = []

    if not available_dates:
        return []

    rows: list[dict] = []
    for target in available_dates:
        if start and target < start:
            continue
        if end and target > end:
            continue
        try:
            rows.extend(load_bet_history(target_date=target.isoformat()) or [])
        except Exception as e:
            log.warning("Erreur load_bet_history(%s) : %s", target.isoformat(), e)

    return rows


def _history_item_from_match(match: dict, item_type: str) -> dict:
    summary = serialize_match_summary(match)
    score = _score_label(summary)
    result_selection = summary.get("result_selection") or {}
    over25_result = _check_over25_result(match)
    winner_result = _check_winner_result(match)

    if item_type == "smart-over25":
        status = _history_status(over25_result)
        selection = {
            "type": "over25",
            "pick": "+2.5",
            "label": "+2,5 buts",
            "probability": summary.get("probabilities", {}).get("over_25"),
        }
    else:
        status = _history_status(winner_result)
        selection = {
            "type": result_selection.get("type") or "single",
            "pick": result_selection.get("pick") or _winner_to_pick(summary.get("predicted_winner")),
            "label": result_selection.get("label") or _winner_to_label(summary.get("predicted_winner")),
            "probability": result_selection.get("probability") or summary.get("winner_proba"),
            "confidence": result_selection.get("confidence") or "faible",
        }

    return {
        "type": item_type,
        "fixture_id": summary["fixture_id"],
        "date": summary["date"],
        "league": summary["league"],
        "home_team": summary["home_team"],
        "away_team": summary["away_team"],
        "score": summary["score"],
        "final_score": score.get("score", ""),
        "result_label": score.get("result_label") or score.get("label", ""),
        "goals_label": score.get("label", ""),
        "status": status,
        "selection": selection,
        "probabilities": summary["probabilities"],
        "is_smart_bet": summary["is_smart_bet"],
        "label": summary["label"],
        "result_selection": result_selection,
    }


def _winner_to_pick(value: str) -> str:
    return {"home": "1", "draw": "N", "away": "2"}.get(str(value or "").lower(), "")


def _winner_to_label(value: str) -> str:
    return {
        "home": "Victoire domicile",
        "draw": "Match nul",
        "away": "Victoire extérieur",
    }.get(str(value or "").lower(), "")


def _is_real_smart_over25(match: dict) -> bool:
    summary = serialize_match_summary(match)
    label = str(summary.get("label") or "").upper()
    return bool(summary.get("is_smart_bet")) and ("OVER" in label or "+2" in label or "2.5" in label)


def _merge_supabase_and_local(target_date: _date) -> list[dict]:
    """Charge depuis Supabase (source de vérité) + cache local (fallback) avec dedup."""
    iso = target_date.isoformat()
    try:
        supabase_data = load_bet_history(target_date=iso) or []
    except Exception as e:
        log.warning("Erreur load_bet_history(%s) : %s", iso, e)
        supabase_data = []

    local_data = load_daily_cache(target_date) or []

    seen = {str(m.get("fixture_id")) for m in supabase_data}
    merged = list(supabase_data)
    for m in local_data:
        fid = str(m.get("fixture_id", ""))
        if fid and fid not in seen:
            merged.append(m)
            seen.add(fid)
    return merged


# ══════════════════════════════════════════════════════════════
# GET /api/history
# ══════════════════════════════════════════════════════════════
@router.get("", summary="Historique réel des analyses enregistrées")
@router.get("/", summary="Historique réel des analyses enregistrées")
async def list_history(
    type: Optional[str] = Query(None, description="smart-over25, smart-result ou result"),
    from_date: Optional[str] = Query(None, alias="from", description="Date début YYYY-MM-DD"),
    to_date: Optional[str] = Query(None, alias="to", description="Date fin YYYY-MM-DD"),
):
    start = _parse_date(from_date) if from_date else None
    end = _parse_date(to_date) if to_date else None

    raw_matches = _load_real_history(start=start, end=end)
    items: list[dict] = []
    type_filter = type or None

    for match in raw_matches:
        if not _is_after_history_start(match):
            continue
        if not _date_in_range(match.get("date", ""), start, end):
            continue
        summary = serialize_match_summary(match)
        result_selection = summary.get("result_selection") or {}

        if _is_real_smart_over25(match) and type_filter in (None, "smart-over25"):
            items.append(_history_item_from_match(match, "smart-over25"))

        if result_selection.get("is_result_selection") and type_filter in (None, "result"):
            items.append(_history_item_from_match(match, "result"))

        # Aucune règle officielle Smart Sim Résultat n'est validée côté backend.
        # On renvoie donc vide pour ce type au lieu d'inventer un historique.
        if type_filter == "smart-result":
            continue

    items.sort(key=lambda item: item.get("date") or "", reverse=True)
    if type_filter == "smart-result":
        items = []

    return {
        "count": len(items),
        "items": items,
        "source": "supabase.bet_history",
        "types": {
            "smart_over25": len([item for item in items if item["type"] == "smart-over25"]),
            "smart_result": len([item for item in items if item["type"] == "smart-result"]),
            "result": len([item for item in items if item["type"] == "result"]),
        },
    }


# ══════════════════════════════════════════════════════════════
# GET /api/history/dates
# ══════════════════════════════════════════════════════════════
@router.get("/dates", summary="Liste des dates disponibles dans l'historique")
async def list_history_dates():
    """
    Renvoie toutes les dates pour lesquelles on a des données
    (Supabase bet_history + cache local fusionnés).
    """
    try:
        supabase_dates = list_bet_history_dates() or []
    except Exception as e:
        log.warning("Erreur list_bet_history_dates : %s", e)
        supabase_dates = []

    local_dates = list_cached_dates() or []
    all_dates = sorted({*supabase_dates, *local_dates}, reverse=True)

    return {
        "count": len(all_dates),
        "dates": [d.isoformat() for d in all_dates],
        "sources": {
            "supabase": len(supabase_dates),
            "local_cache": len(local_dates),
        },
    }


# ══════════════════════════════════════════════════════════════
# GET /api/history/{date}
# ══════════════════════════════════════════════════════════════
@router.get("/{date_str}", summary="Tous les matchs d'une date passée")
async def get_history_for_date(date_str: str):
    """
    Retourne tous les matchs d'une date (Supabase + cache local fusionnés).
    Inclut le résultat réel et la classification win/loss pour Over 2.5 et 1X2.
    """
    target = _parse_date(date_str)
    matches = _merge_supabase_and_local(target)
    matches = [match for match in matches if _is_after_history_start(match)]

    if not matches:
        return {
            "date": target.isoformat(),
            "count": 0,
            "matches": [],
            "hint": "Aucune donnée pour cette date.",
        }

    enriched = []
    for m in matches:
        item = serialize_match_summary(m)
        item["over25_result"] = _check_over25_result(m)
        item["winner_result"] = _check_winner_result(m)
        enriched.append(item)

    return {
        "date": target.isoformat(),
        "count": len(enriched),
        "matches": enriched,
    }


# ══════════════════════════════════════════════════════════════
# GET /api/history/smart/{date}
# ══════════════════════════════════════════════════════════════
@router.get(
    "/smart/{date_str}",
    summary="Smart Selections d'une date + bilan win/loss",
)
async def get_smart_history(date_str: str):
    """
    Bilan des Smart Selections pour une date donnée :
    - filtre Smart Sim uniquement
    - calcule wins / losses / pending sur Over 2.5
    - retourne winrate
    """
    target = _parse_date(date_str)
    all_matches = _merge_supabase_and_local(target)
    all_matches = [match for match in all_matches if _is_after_history_start(match)]
    smart = [m for m in all_matches if _is_smart(m)]

    if not smart:
        return {
            "date": target.isoformat(),
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "winrate": 0.0,
            "matches": [],
        }

    wins, losses, pending = 0, 0, 0
    enriched = []
    for m in smart:
        result = _check_over25_result(m)
        if result in {"win", "won"}:
            wins += 1
        elif result in {"loss", "lost"}:
            losses += 1
        else:
            pending += 1
        item = serialize_match_summary(m)
        item["over25_result"] = result
        enriched.append(item)

    decided = wins + losses
    winrate = round(wins / decided * 100, 2) if decided > 0 else 0.0

    return {
        "date": target.isoformat(),
        "count": len(smart),
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "winrate": winrate,
        "total_analyzed": len(all_matches),
        "matches": enriched,
    }
