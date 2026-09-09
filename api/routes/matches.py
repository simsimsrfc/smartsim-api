"""
Routes /api/matches/*

GET /api/matches/today                 — Matchs du jour
GET /api/matches/smart-selections      — Smart Sim (filtré par seuil)
GET /api/matches/{fixture_id}          — Fiche match détaillée
"""

import logging
import os
from datetime import date as _date, datetime as _datetime, timedelta
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _TZ_PARIS = ZoneInfo("Europe/Paris")
except ImportError:
    _TZ_PARIS = None

from fastapi import APIRouter, HTTPException, Query, Header

from data_fetcher import (
    load_daily_cache,
    save_daily_cache,
    fetch_all_today,
    fetch_all_by_date,
    fetch_fixtures_by_date,
    list_cached_dates,
    get_quota_status,
)
import os as _os, logging as _logging
from model import predict_today as _predict_today_legacy, SimbetEnsemble

_ENGINE = _os.environ.get("SIMBET_ENGINE", "v2").lower()
if _ENGINE == "v2":
    try:
        from simbet_v2_bridge import predict_today_v2 as predict_today
        _logging.getLogger("SmartSim.api").info(
            "[engine] SimBet V2 bridge active (matches route)")
    except Exception as _exc:
        _logging.getLogger("SmartSim.api").warning(
            "[engine] V2 bridge unavailable, using legacy: %s", _exc)
        predict_today = _predict_today_legacy
else:
    predict_today = _predict_today_legacy
from config import LEAGUES, SMART_BET_THRESHOLD
from extra_markets import compute_extra_markets
from supabase_db import load_bet_history, list_bet_history_dates, save_bet_history

from api.serializers import serialize_match_summary, serialize_match_detail

log = logging.getLogger("SmartSim.api.matches")
router = APIRouter()


# ──────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────
def _load_today_data() -> list[dict]:
    """Charge le cache du jour, ou liste vide."""
    return load_daily_cache(_paris_today()) or []


def _paris_today() -> _date:
    """Date courante selon Europe/Paris."""
    if _TZ_PARIS is not None:
        return _datetime.now(_TZ_PARIS).date()
    return _date.today()


def _parse_iso_date(value: str) -> _date:
    try:
        return _date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        raise HTTPException(400, "Paramètre date invalide. Format attendu : YYYY-MM-DD.")


def _resolve_target_date(
    day: Optional[str] = None,
    date_value: Optional[str] = None,
) -> _date:
    today = _paris_today()
    if date_value:
        return _parse_iso_date(date_value)
    if not day or day == "today":
        return today
    if day == "tomorrow":
        return today + timedelta(days=1)
    raise HTTPException(400, "Paramètre day invalide. Valeurs attendues : today ou tomorrow.")


def _has_extra_markets(match: dict) -> bool:
    """Vérifie que les marchés complémentaires historiques sont présents."""
    extra = match.get("_extra") or {}
    required = (
        "proba_o15",
        "proba_btts",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "winner",
        "winner_proba",
    )
    return all(key in extra and extra.get(key) is not None for key in required)


def _enrich_with_extra_markets(results: list[dict]) -> list[dict]:
    """
    Injecte les marchés complémentaires de l'ancien moteur SIMBET.
    Ne reconstruit pas les features : il réutilise celles produites par predict_today().
    """
    enriched = []
    for result in results:
        entry = result
        if _has_extra_markets(entry):
            enriched.append(entry)
            continue

        features = entry.get("features") or {}
        pred = entry.get("prediction") or {}
        try:
            proba_o25 = float(pred.get("proba_over25"))
        except (TypeError, ValueError):
            log.warning("Impossible de calculer _extra pour fixture %s : proba_o25 absente", entry.get("fixture_id"))
            enriched.append(entry)
            continue

        try:
            entry["_extra"] = compute_extra_markets(
                features,
                proba_o25,
                entry.get("league_id"),
            )
        except Exception as e:
            log.warning("Erreur compute_extra_markets fixture %s : %s", entry.get("fixture_id"), e)

        enriched.append(entry)

    return enriched


def _persist_bet_history(results: list[dict], target_date: _date) -> int:
    """Sauvegarde les analyses pré-match enrichies dans bet_history."""
    try:
        return save_bet_history(results, target_date=target_date.isoformat())
    except Exception as e:
        log.warning("save_bet_history(%s) : %s", target_date.isoformat(), e)
        return 0


def _persist_historical_data(results: list[dict]) -> int:
    """
    Étape 21.6 : pousse les matchs analysés J/J+1 dans `historical_data` Supabase
    via upsert_matches() (UPSERT inconditionnel, sans dedup local_cache).

    Permet d'enrichir les fixtures déjà connues avec les nouveaux champs
    (odds_data.extended_markets, all_bookmakers_raw, odds_fetched_at, etc.).
    """
    try:
        from supabase_db import upsert_matches
        matches_dict = {}
        for r in results:
            fid = str(r.get("fixture_id", ""))
            if not fid:
                continue
            matches_dict[fid] = r
        if matches_dict:
            return upsert_matches(matches_dict)
        return 0
    except Exception as e:
        log.warning("upsert_matches J/J+1 : %s", e)
        return 0


def _check_refresh_auth(x_cron_secret: Optional[str]) -> None:
    """
    Étape 22.8 — Protection optionnelle des endpoints refresh=true.

    Si la variable d'environnement CRON_SECRET est définie :
      - le header X-Cron-Secret doit correspondre exactement, sinon HTTP 403.
    Si CRON_SECRET n'est pas défini :
      - comportement actuel conservé (refresh public).

    Cette protection s'applique uniquement aux appels avec refresh=true.
    Les lectures simples (sans refresh) restent publiques inchangées.
    """
    expected = (os.environ.get("CRON_SECRET") or "").strip()
    if not expected:
        return   # public par défaut, rétrocompat
    if (x_cron_secret or "").strip() != expected:
        raise HTTPException(403, "Forbidden: invalid or missing X-Cron-Secret header")


def _load_matches_for_date(target_date: _date, refresh: bool = False,
                           max_matches: Optional[int] = None) -> dict:
    """Charge ou calcule les analyses d'une date sans appel API si refresh=False."""
    cached = None if refresh else load_daily_cache(target_date)
    if cached:
        if not all(_has_extra_markets(m) for m in cached):
            cached = _enrich_with_extra_markets(cached)
            save_daily_cache(cached, target_date=target_date)
            _persist_bet_history(cached, target_date)
        return {
            "date": target_date.isoformat(),
            "source": "cache",
            "count": len(cached),
            "matches": [serialize_match_summary(m) for m in cached],
        }

    if not refresh:
        return {
            "date": target_date.isoformat(),
            "source": "empty",
            "count": 0,
            "matches": [],
            "hint": "Aucun cache. Appelle ?refresh=true pour fetcher.",
        }

    # V2 bridge active, no legacy model needed
    if not SimbetEnsemble.exists() and _ENGINE != "v2":
        raise HTTPException(503, "Modèle ML non entraîné.")

    try:
        if max_matches is not None:
            raw = fetch_all_by_date(target_date, max_matches=max_matches)
        else:
            raw = fetch_all_by_date(target_date)
    except Exception as e:
        log.error("fetch_all_by_date(%s) : %s", target_date.isoformat(), e)
        raise HTTPException(502, f"Erreur API-Football : {e}")

    if not raw:
        return {
            "date": target_date.isoformat(),
            "source": "live",
            "count": 0,
            "matches": [],
            "hint": "Aucun match dans les ligues configurées pour cette date.",
            "quota": get_quota_status(),
        }

    results = _enrich_with_extra_markets(predict_today(raw))
    save_daily_cache(results, target_date=target_date)
    persisted = _persist_bet_history(results, target_date)
    # Étape 21.3 : push durable vers historical_data Supabase
    historical_saved = _persist_historical_data(results)
    return {
        "date": target_date.isoformat(),
        "source": "live",
        "count": len(results),
        "matches": [serialize_match_summary(m) for m in results],
        "bet_history_saved": persisted,
        "historical_data_saved": historical_saved,
        "quota": get_quota_status(),
    }


def _is_smart_selection(m: dict) -> bool:
    """Critère Smart Sim : uniquement un match déjà marqué Smart Sim."""
    pred = m.get("prediction", {}) or {}
    smart = pred.get("smart_bet", {}) or {}
    return bool(m.get("is_smart_bet")) or bool(smart.get("is_smart_bet"))


def _over25_probability(m: dict) -> float:
    """Retourne la proba O2.5 au format 0-1, quel que soit le format source."""
    pred = m.get("prediction", {}) or {}
    probabilities = m.get("probabilities", {}) or {}
    try:
        return float(pred.get("proba_over25") or probabilities.get("over_25") or 0)
    except (TypeError, ValueError):
        return 0.0


def _fixture_id_from_api_fixture(fixture: dict) -> str:
    return str((fixture.get("fixture") or {}).get("id") or "")


def _cached_fixture_id(match: dict) -> str:
    return str(match.get("fixture_id") or "")


def _is_analyzed_match(match: dict) -> bool:
    pred = match.get("prediction") or {}
    return bool(_cached_fixture_id(match) and pred.get("proba_over25") is not None and _has_extra_markets(match))


def _fetch_fixture_ids_for_date(target_date: _date) -> dict:
    """Récupère uniquement les fixtures API-Football configurées pour une date."""
    fixtures_by_league = {}
    fixture_ids = []

    for league_id, meta in LEAGUES.items():
        try:
            fixtures = fetch_fixtures_by_date(league_id, meta["season"], target_date) or []
        except Exception as e:
            log.warning("coverage fetch_fixtures_by_date(%s, %s) : %s", league_id, target_date.isoformat(), e)
            fixtures = []

        ids = [_fixture_id_from_api_fixture(fixture) for fixture in fixtures]
        ids = [fid for fid in ids if fid]
        if ids:
            fixtures_by_league[str(league_id)] = {
                "name": meta.get("name"),
                "count": len(ids),
                "fixture_ids": ids,
            }
            fixture_ids.extend(ids)

    return {
        "fixture_ids": sorted(set(fixture_ids)),
        "fixtures_by_league": fixtures_by_league,
    }


def _load_sample_source_matches() -> tuple[Optional[str], list[dict]]:
    """Charge les mêmes sources que /sample : bet_history Supabase + cache local."""
    try:
        sb_dates = list_bet_history_dates() or []
    except Exception as e:
        log.warning("list_bet_history_dates : %s", e)
        sb_dates = []

    local_dates = list_cached_dates() or []
    all_dates = sorted({*sb_dates, *local_dates}, reverse=True)

    if not all_dates:
        return None, []

    target = all_dates[0]
    iso = target.isoformat()

    try:
        sb_data = load_bet_history(target_date=iso) or []
    except Exception as e:
        log.warning("load_bet_history(%s) : %s", iso, e)
        sb_data = []

    local_data = load_daily_cache(target) or []

    seen = {str(m.get("fixture_id")) for m in sb_data}
    merged = list(sb_data)
    for m in local_data:
        fid = str(m.get("fixture_id", ""))
        if fid and fid not in seen:
            merged.append(m)
            seen.add(fid)

    return iso, merged


def _find_match_in_recent_local_cache(fid: str) -> Optional[dict]:
    """Cherche un fixture_id dans le cache du jour puis les caches locaux récents."""
    today_data = _load_today_data()
    found = next(
        (m for m in today_data if str(m.get("fixture_id")) == fid),
        None,
    )
    if found is not None:
        return found

    for d in list_cached_dates()[:14]:
        data = load_daily_cache(d) or []
        found = next(
            (m for m in data if str(m.get("fixture_id")) == fid),
            None,
        )
        if found:
            log.info("Match %s trouvé dans le cache du %s", fid, d)
            return found

    return None


def _find_match_in_sample_sources(fid: str) -> Optional[dict]:
    """
    Cherche un fixture_id dans les mêmes sources que /sample :
    Supabase bet_history, puis cache local de la date concernée.
    """
    try:
        sb_dates = list_bet_history_dates() or []
    except Exception as e:
        log.warning("list_bet_history_dates : %s", e)
        sb_dates = []

    local_dates = list_cached_dates() or []
    all_dates = sorted({*sb_dates, *local_dates}, reverse=True)

    for target in all_dates:
        iso = target.isoformat()

        try:
            sb_data = load_bet_history(target_date=iso) or []
        except Exception as e:
            log.warning("load_bet_history(%s) : %s", iso, e)
            sb_data = []

        found = next(
            (m for m in sb_data if str(m.get("fixture_id")) == fid),
            None,
        )
        if found:
            log.info("Match %s trouvé dans bet_history Supabase du %s", fid, iso)
            return found

        local_data = load_daily_cache(target) or []
        found = next(
            (m for m in local_data if str(m.get("fixture_id")) == fid),
            None,
        )
        if found:
            log.info("Match %s trouvé dans le cache sample du %s", fid, iso)
            return found

    return None


# ══════════════════════════════════════════════════════════════
# GET /api/matches/today
# ══════════════════════════════════════════════════════════════
@router.get("/today", summary="Matchs du jour")
async def list_matches_today(
    refresh: bool = Query(False, description="Si True : refetch via API-Football"),
    limit:   Optional[int] = Query(None, ge=1, le=200,
                                   description="Plafond matchs analysés (optionnel)"),
    x_cron_secret: Optional[str] = Header(default=None, alias="X-Cron-Secret"),
):
    if refresh:
        _check_refresh_auth(x_cron_secret)
    return _load_matches_for_date(
        _paris_today(),
        refresh=refresh,
        max_matches=limit,
    )


# ══════════════════════════════════════════════════════════════
# GET /api/matches/tomorrow
# ══════════════════════════════════════════════════════════════
@router.get("/tomorrow", summary="Matchs de demain")
async def list_matches_tomorrow(
    refresh: bool = Query(False, description="Si True : refetch via API-Football"),
    limit:   int  = Query(30, ge=1, le=200,
                          description="Plafond matchs analysés (default 30 pour J+1)"),
    x_cron_secret: Optional[str] = Header(default=None, alias="X-Cron-Secret"),
):
    if refresh:
        _check_refresh_auth(x_cron_secret)
    return _load_matches_for_date(
        _paris_today() + timedelta(days=1),
        refresh=refresh,
        max_matches=limit,
    )


# ══════════════════════════════════════════════════════════════
# GET /api/matches/date?date=YYYY-MM-DD
# ══════════════════════════════════════════════════════════════
@router.get("/date", summary="Matchs d'une date explicite")
async def list_matches_by_date(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    refresh: bool = Query(False, description="Si True : refetch via API-Football"),
    x_cron_secret: Optional[str] = Header(default=None, alias="X-Cron-Secret"),
):
    if refresh:
        _check_refresh_auth(x_cron_secret)
    return _load_matches_for_date(_parse_iso_date(date), refresh=refresh)


# ══════════════════════════════════════════════════════════════
# GET /api/matches/coverage
# ══════════════════════════════════════════════════════════════
@router.get("/coverage", summary="Diagnostic de complétude fixtures/cache")
async def matches_coverage(
    day: Optional[str] = Query(None, description="today ou tomorrow. Par défaut : today"),
    date: Optional[str] = Query(None, description="Date explicite au format YYYY-MM-DD"),
):
    target_date = _resolve_target_date(day=day, date_value=date)
    api_data = _fetch_fixture_ids_for_date(target_date)
    api_ids = set(api_data["fixture_ids"])

    cached = load_daily_cache(target_date) or []
    cache_ids = {_cached_fixture_id(match) for match in cached if _cached_fixture_id(match)}
    analyzed_ids = {_cached_fixture_id(match) for match in cached if _is_analyzed_match(match)}
    missing_ids = sorted(api_ids - cache_ids)
    unanalyzed_ids = sorted(cache_ids - analyzed_ids)

    return {
        "date": target_date.isoformat(),
        "fixtures_api_count": len(api_ids),
        "cache_count": len(cache_ids),
        "analyzed_count": len(analyzed_ids),
        "missing_count": len(missing_ids),
        "unanalyzed_count": len(unanalyzed_ids),
        "cache_complete": len(missing_ids) == 0 and len(unanalyzed_ids) == 0 and len(api_ids) == len(cache_ids),
        "fixture_ids_missing": missing_ids,
        "fixture_ids_unanalyzed": unanalyzed_ids,
        "fixture_ids_present": sorted(cache_ids),
        "fixture_ids_api": sorted(api_ids),
        "fixtures_by_league": api_data["fixtures_by_league"],
        "quota": get_quota_status(),
    }


# ══════════════════════════════════════════════════════════════
# GET /api/matches/sample
# ══════════════════════════════════════════════════════════════
@router.get(
    "/sample",
    summary="Matchs d'exemple — pioche la dernière date dispo dans Supabase (0 appel API)",
)
async def list_sample_matches():
    """
    Retourne les matchs de la date la plus récente disponible en base
    (Supabase bet_history + cache local fusionnés). N'appelle PAS API-Football,
    n'utilise PAS le modèle ML. Sert au frontend pour tester avec de vraies
    données déjà calculées quand /today est vide.

    Format de réponse identique à /today.
    """
    # 1. Chercher la dernière date avec données
    iso, merged = _load_sample_source_matches()

    if not iso:
        return {
            "date": None,
            "source": "empty",
            "count": 0,
            "matches": [],
            "hint": "Aucune date disponible dans Supabase ni dans le cache local.",
        }

    return {
        "date": iso,
        "source": "supabase-sample",
        "count": len(merged),
        "matches": [serialize_match_summary(m) for m in merged],
    }


# ══════════════════════════════════════════════════════════════
# GET /api/matches/smart-selections
# ══════════════════════════════════════════════════════════════
@router.get("/smart-selections", summary="Sélections Smart Sim du jour")
async def list_smart_selections(
    min_proba: float = Query(
        0.0, ge=0.0, le=1.0,
        description="Filtre additionnel : proba Over 2.5 minimale",
    ),
    day: Optional[str] = Query(
        None,
        description="today ou tomorrow. Par défaut : today",
    ),
    date: Optional[str] = Query(
        None,
        description="Date explicite au format YYYY-MM-DD",
    ),
):
    """
    Retourne uniquement les matchs jugés 'Smart Selections' :
    - match déjà marqué Smart Sim par le moteur
    - filtrage optionnel sur la proba Over 2.5
    """
    target_date = _resolve_target_date(day=day, date_value=date)
    date_label = target_date.isoformat()
    source = "cache"
    matches = load_daily_cache(target_date) or []
    smart = [m for m in matches if _is_smart_selection(m)]

    if min_proba > 0:
        smart = [
            m for m in smart
            if _over25_probability(m) >= min_proba
        ]

    smart.sort(
        key=_over25_probability,
        reverse=True,
    )

    return {
        "date": date_label,
        "source": source,
        "threshold": SMART_BET_THRESHOLD,
        "min_proba_filter": min_proba,
        "count": len(smart),
        "total_matches": len(matches),
        "matches": [serialize_match_summary(m) for m in smart],
    }


# ══════════════════════════════════════════════════════════════
# GET /api/matches/debug/compare-models — v1 vs v2 side by side
# ══════════════════════════════════════════════════════════════
@router.get("/debug/compare-models", summary="Compare Poisson v1 vs v2 (debug)")
async def debug_compare_models(
    day: Optional[str] = Query("today", description="today ou tomorrow"),
):
    """Retourne pour chaque match du cache : {market, poisson_v1, poisson_v2, delta}."""
    from simbet_v2_bridge import _priors_from_form, _priors_from_form_v2, _implied, _normalize
    target_date = _resolve_target_date(day=day)
    matches = load_daily_cache(target_date) or []
    out = []
    for m in matches:
        odds = m.get("odds") or {}
        ext = odds.get("extended_markets") or {}
        p_h, p_d, p_a = _normalize((
            _implied(ext.get("home")), _implied(ext.get("draw")), _implied(ext.get("away"))))
        v1 = _priors_from_form(m)
        v2 = _priors_from_form_v2(m)
        out.append({
            "match": f"{m['home_team']['name']} vs {m['away_team']['name']}",
            "market":  {"H": round(p_h, 3), "D": round(p_d, 3), "A": round(p_a, 3),
                          "O25": round(_implied(ext.get("over_25")), 3)},
            "v1": {"H": round(v1["home"], 3), "D": round(v1["draw"], 3), "A": round(v1["away"], 3),
                     "O25": round(v1["over_25"], 3), "BTTS": round(v1["btts"], 3)},
            "v2": {"H": round(v2["home"], 3), "D": round(v2["draw"], 3), "A": round(v2["away"], 3),
                     "O25": round(v2["over_25"], 3), "BTTS": round(v2["btts"], 3),
                     "lam_h": round(v2.get("lam_home", 0), 2), "lam_a": round(v2.get("lam_away", 0), 2)},
            "delta_H": round(v2["home"] - v1["home"], 3),
            "delta_O25": round(v2["over_25"] - v1["over_25"], 3),
            "rest_h": m.get("home_rest_days"),
            "rest_a": m.get("away_rest_days"),
            "inj_h": m.get("home_missing_weighted"),
            "inj_a": m.get("away_missing_weighted"),
        })
    return {"date": target_date.isoformat(), "count": len(out), "matches": out}


# ══════════════════════════════════════════════════════════════
# POST /api/matches/sync-results — worker post-match (cron only)
# ══════════════════════════════════════════════════════════════
@router.post("/sync-results", summary="Synchronise les résultats des matchs terminés")
async def sync_results_endpoint(
    max_api_calls: int = Query(50, ge=1, le=200, description="Plafond strict d'appels API par run"),
    days_back:     int = Query(7,  ge=1, le=30, description="Fenêtre rétrospective en jours"),
    x_cron_secret: Optional[str] = Header(default=None, alias="X-Cron-Secret"),
):
    """
    Worker de synchronisation post-match.

    Étapes :
      1. SELECT bet_history WHERE date in [today - days_back, today]
         AND (resolved_at IS NULL OR match_status non-final)
      2. Pour chaque fixture : GET /fixtures?id=X (1 call API)
      3. Si statut final (FT/AET/PEN) → UPDATE results_*_won + total_goals + actual_winner
      4. Si statut terminal (CANC/PST/...) → UPDATE match_status, marque resolved_at
      5. Si statut intermédiaire (LIVE/HT/NS) → UPDATE status + scores partiels

    Protégé par X-Cron-Secret (si CRON_SECRET défini en env).
    Aucun retraining, aucun modèle touché.
    """
    _check_refresh_auth(x_cron_secret)

    from api.results_sync import sync_pending_results

    try:
        report = sync_pending_results(max_api_calls=max_api_calls, days_back=days_back)
    except Exception as e:
        log.error("sync-results failure: %s", e)
        raise HTTPException(500, f"sync-results failed: {str(e)[:200]}")

    return report


# ══════════════════════════════════════════════════════════════
# GET /api/matches/{fixture_id}
# ══════════════════════════════════════════════════════════════
@router.get(
    "/{fixture_id}",
    summary="Fiche match détaillée (forme + H2H + analyse)",
)
async def get_match_detail(fixture_id: str):
    """
    Cherche le match par fixture_id :
    1. D'abord dans le cache du jour
    2. Sinon dans les caches journaliers récents (jusqu'à 14 jours)
    3. Sinon dans les sources utilisées par /sample (Supabase + cache local)
    """
    fid = str(fixture_id)

    found = _find_match_in_recent_local_cache(fid)
    if found is None:
        found = _find_match_in_sample_sources(fid)

    if found is None:
        raise HTTPException(
            404,
            f"Match {fid} introuvable dans les caches récents et les sources sample.",
        )

    return serialize_match_detail(found)
