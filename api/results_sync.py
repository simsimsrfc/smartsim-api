"""
Synchronisation post-match des résultats (Phase 1 autonome).

Ce module fournit :
  - `compute_prediction_results` : fonction PURE qui calcule win/loss à partir
    d'un score (home_goals, away_goals) et d'une prédiction de bet_history.
  - `fetch_fixture_final_status` : appelle l'API-Football pour un fixture_id
    et retourne (status_code, home_goals, away_goals).
  - `sync_pending_results` : orchestre la sync — sélectionne les fixtures
    bet_history non résolus, fetch côté API, met à jour Supabase.

Toutes les écritures Supabase sont UPDATE ciblés par fixture_id.
Pas de bulk replace. Pas d'effet sur les colonnes pré-match (proba_*, odds_*).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger("SmartSim.results_sync")

# Statuts API-Football considérés comme "terminé" (score final figé)
FINAL_STATUSES = {"FT", "AET", "PEN"}

# Statuts terminaux non-résolvables (pas de score à enregistrer)
TERMINAL_NON_RESOLVED = {"CANC", "PST", "ABD", "AWD", "WO"}


def compute_prediction_results(
    home_goals: Optional[int],
    away_goals: Optional[int],
    predicted_winner: Optional[str],
) -> dict[str, Any]:
    """
    Fonction PURE : calcule win/loss par marché à partir d'un score réel.

    Aucun appel I/O. Aucun side effect. Retourne un dict prêt à être inséré
    dans bet_history via UPDATE.

    Parameters
    ----------
    home_goals : int | None
        Buts marqués par l'équipe domicile (None si match non terminé).
    away_goals : int | None
        Buts marqués par l'équipe extérieure.
    predicted_winner : str | None
        Valeur de la colonne `winner` dans bet_history : "home" / "draw" / "away"
        (ou variantes nulles).

    Returns
    -------
    dict
        {
          "total_goals":         int | None,
          "actual_winner":       "home" | "draw" | "away" | None,
          "result_over25_won":   bool | None,
          "result_over15_won":   bool | None,
          "result_btts_won":     bool | None,
          "result_winner_won":   bool | None,
        }
        Toutes les valeurs sont None si home_goals ou away_goals est None
        (score indisponible / match non terminé).
    """
    if home_goals is None or away_goals is None:
        return {
            "total_goals":        None,
            "actual_winner":      None,
            "result_over25_won":  None,
            "result_over15_won":  None,
            "result_btts_won":    None,
            "result_winner_won":  None,
        }

    hg = int(home_goals)
    ag = int(away_goals)
    total = hg + ag

    if hg > ag:
        actual = "home"
    elif ag > hg:
        actual = "away"
    else:
        actual = "draw"

    pred_norm = (predicted_winner or "").strip().lower() or None
    # Normalisation tolérante : "home"/"away"/"draw" attendus, mais on accepte
    # aussi "1"/"2"/"N" au cas où.
    pred_map = {"1": "home", "2": "away", "n": "draw"}
    if pred_norm in pred_map:
        pred_norm = pred_map[pred_norm]

    result_winner_won = (
        bool(pred_norm) and pred_norm in ("home", "draw", "away") and pred_norm == actual
    ) if pred_norm else None

    return {
        "total_goals":        total,
        "actual_winner":      actual,
        "result_over25_won":  total > 2,
        "result_over15_won":  total > 1,
        "result_btts_won":    hg > 0 and ag > 0,
        "result_winner_won":  result_winner_won,
    }


# ──────────────────────────────────────────────────────────────────────────
# I/O — appel API-Football + écriture Supabase
# ──────────────────────────────────────────────────────────────────────────
def fetch_fixture_final_status(fixture_id: str) -> Optional[dict[str, Any]]:
    """
    Récupère le statut final d'un fixture depuis API-Football.
    1 appel API consommé par invocation.

    Returns
    -------
    dict | None
        {
          "fixture_id":  str,
          "status_code": str ("FT", "AET", "PEN", "NS", "LIVE", "PST", ...),
          "home_goals":  int | None,
          "away_goals":  int | None,
          "elapsed":     int | None,
        }
        None si l'appel API échoue ou retourne 0 résultats.
    """
    from data_fetcher import api_get, CACHE_TTL_FIXTURES, QuotaExceeded

    try:
        data = api_get("fixtures", {"id": int(fixture_id)},
                        ttl=300)  # TTL court : 5 min — résultats à jour
    except QuotaExceeded:
        log.warning("Quota épuisé pendant fetch_fixture_final_status(%s)", fixture_id)
        raise
    except Exception as e:
        log.warning("fetch_fixture_final_status(%s) error: %s", fixture_id, e)
        return None

    if not data or not data.get("response"):
        return None

    fx = data["response"][0]
    status = (fx.get("fixture") or {}).get("status") or {}
    goals = fx.get("goals") or {}

    return {
        "fixture_id":  str(fixture_id),
        "status_code": status.get("short", "NS"),
        "home_goals":  goals.get("home"),
        "away_goals":  goals.get("away"),
        "elapsed":     status.get("elapsed"),
    }


def select_pending_fixtures(client, limit: int = 50, days_back: int = 7) -> list[dict]:
    """
    Récupère depuis Supabase les fixtures bet_history qui :
      - sont datés des `days_back` derniers jours
      - ne sont PAS encore résolus (resolved_at IS NULL OU match_status non-final)

    Tri DESC par date (matchs récents d'abord). Limit configurable.
    """
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    today = date.today().isoformat()

    try:
        # PostgREST : OR via .or_("resolved_at.is.null,match_status.not.in.(...)")
        # Plus simple : on fait 2 queries et on dédoublonne côté Python.
        unresolved = client.table("bet_history").select(
            "fixture_id,date,match_status,home_team,away_team,winner,resolved_at"
        ).gte("date", cutoff).lte("date", today).is_("resolved_at", "null").limit(limit).execute()

        rows = list(unresolved.data or [])

        # Ajouter les fixtures avec resolved_at mais status non final (peut arriver
        # si on a synchronisé un match LIVE qui n'est pas encore FT)
        if len(rows) < limit:
            already_resolved = client.table("bet_history").select(
                "fixture_id,date,match_status,home_team,away_team,winner,resolved_at"
            ).gte("date", cutoff).lte("date", today).not_.is_("resolved_at", "null").in_(
                "match_status", ["NS", "1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]
            ).limit(limit - len(rows)).execute()
            seen = {r["fixture_id"] for r in rows}
            for r in (already_resolved.data or []):
                if r["fixture_id"] not in seen:
                    rows.append(r)

        return rows
    except Exception as e:
        log.error("select_pending_fixtures Supabase error: %s", e)
        return []


def update_fixture_results(client, fixture_id: str, status_code: str,
                            home_goals: Optional[int], away_goals: Optional[int],
                            predicted_winner: Optional[str]) -> bool:
    """
    Met à jour la ligne bet_history pour ce fixture avec :
      - match_status
      - home_goals / away_goals
      - colonnes result_*_won
      - total_goals, actual_winner
      - resolved_at

    Returns True si UPDATE réussi.
    """
    results = compute_prediction_results(home_goals, away_goals, predicted_winner)

    update_payload = {
        "match_status":       status_code,
        "home_goals":         home_goals,
        "away_goals":         away_goals,
        "total_goals":        results["total_goals"],
        "actual_winner":      results["actual_winner"],
        "result_over25_won":  results["result_over25_won"],
        "result_over15_won":  results["result_over15_won"],
        "result_btts_won":    results["result_btts_won"],
        "result_winner_won":  results["result_winner_won"],
        "resolved_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Si la migration n'est pas encore appliquée, certains champs n'existent
    # pas → fallback graceful. Liste des colonnes susceptibles d'être absentes.
    NEW_COLS = (
        "result_over25_won", "result_over15_won", "result_btts_won",
        "result_winner_won", "total_goals", "actual_winner", "resolved_at",
    )

    try:
        client.table("bet_history").update(update_payload).eq("fixture_id", fixture_id).execute()
        return True
    except Exception as e:
        msg = str(e)
        if any(c in msg for c in NEW_COLS):
            log.warning("UPDATE bet_history fixture %s : colonnes results_* absentes "
                          "(migration pas appliquée ?) — fallback sans : %s", fixture_id, msg[:150])
            stripped = {k: v for k, v in update_payload.items() if k not in NEW_COLS}
            try:
                client.table("bet_history").update(stripped).eq("fixture_id", fixture_id).execute()
                return True
            except Exception as e2:
                log.error("UPDATE bet_history fixture %s : fallback failed: %s", fixture_id, e2)
                return False
        log.error("UPDATE bet_history fixture %s : %s", fixture_id, msg[:200])
        return False


def sync_pending_results(max_api_calls: int = 50, days_back: int = 7) -> dict[str, Any]:
    """
    Orchestrateur sync-results :
      1. SELECT bet_history fixtures pending
      2. Pour chaque : fetch_fixture_final_status (1 call API)
      3. Si statut final → update_fixture_results
      4. Si statut non-final → met juste à jour match_status sans toucher results_*
      5. Stop si max_api_calls atteint ou quota API limite

    Retourne un résumé pour le worker.
    """
    from supabase_db import _get_client
    from data_fetcher import get_quota_status, QuotaExceeded

    client = _get_client()
    if client is None:
        return {"error": "Supabase indisponible", "synced": 0, "api_calls_used": 0}

    pending = select_pending_fixtures(client, limit=max_api_calls, days_back=days_back)
    if not pending:
        return {"synced": 0, "api_calls_used": 0, "processed": 0,
                "message": "no pending fixtures"}

    quota_before = get_quota_status()
    api_calls = 0
    resolved_final = 0      # synchronisés et statut FT/AET/PEN
    resolved_terminal = 0   # CANC/PST/ABD/AWD (synchronisés mais sans score)
    resolved_intermediate = 0  # statut MAJ mais pas final (LIVE/HT)
    errors: list[dict] = []
    skipped_no_change = 0

    for row in pending:
        if api_calls >= max_api_calls:
            log.info("max_api_calls reached (%d), stopping", max_api_calls)
            break

        fid = str(row.get("fixture_id"))
        try:
            fx = fetch_fixture_final_status(fid)
        except QuotaExceeded:
            errors.append({"fixture_id": fid, "error": "quota_exceeded"})
            break
        except Exception as e:
            errors.append({"fixture_id": fid, "error": str(e)[:120]})
            continue

        api_calls += 1
        if fx is None:
            errors.append({"fixture_id": fid, "error": "fetch_returned_none"})
            continue

        status = fx["status_code"]
        hg = fx["home_goals"]
        ag = fx["away_goals"]
        predicted = row.get("winner")

        if status in FINAL_STATUSES and hg is not None and ag is not None:
            ok = update_fixture_results(client, fid, status, hg, ag, predicted)
            if ok:
                resolved_final += 1
            else:
                errors.append({"fixture_id": fid, "error": "update_failed"})
        elif status in TERMINAL_NON_RESOLVED:
            # Pas de score → on met juste match_status + resolved_at pour ne plus retenter
            ok = update_fixture_results(client, fid, status, None, None, predicted)
            if ok:
                resolved_terminal += 1
            else:
                errors.append({"fixture_id": fid, "error": "update_terminal_failed"})
        else:
            # LIVE / HT / NS : on met juste à jour match_status (pas de resolved_at)
            try:
                client.table("bet_history").update({
                    "match_status": status,
                    "home_goals":   hg,
                    "away_goals":   ag,
                }).eq("fixture_id", fid).execute()
                resolved_intermediate += 1
            except Exception as e:
                errors.append({"fixture_id": fid, "error": f"intermediate_update: {str(e)[:120]}"})

    quota_after = get_quota_status()

    return {
        "processed":            len(pending),
        "api_calls_used":       api_calls,
        "resolved_final":       resolved_final,
        "resolved_terminal":    resolved_terminal,
        "resolved_intermediate": resolved_intermediate,
        "skipped_no_change":    skipped_no_change,
        "errors_count":         len(errors),
        "errors":               errors[:10],
        "quota_before":         quota_before,
        "quota_after":          quota_after,
        "ran_at":               datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
