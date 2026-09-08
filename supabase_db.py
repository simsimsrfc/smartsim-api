"""
Smart Sim — Supabase Database
Source de vérité persistante du projet.
Tables : bet_history, historical_data
Cache disque local pour accélération et fallback offline.
"""

import os
import json
import logging
from datetime import timezone, timedelta
from pathlib import Path
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _TZ_PARIS = ZoneInfo("Europe/Paris")
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        _TZ_PARIS = ZoneInfo("Europe/Paris")
    except ImportError:
        _TZ_PARIS = timezone(timedelta(hours=2))

log = logging.getLogger("SmartSim.supabase_db")

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_CACHE_FILE = _PROJECT_ROOT / "cache" / "historical_cache.json"
_BET_HISTORY_DATES_CACHE = _PROJECT_ROOT / "cache" / "bet_history_dates.json"

_supabase_client = None


def _paris_today_iso() -> str:
    from datetime import datetime as _dt
    return _dt.now(_TZ_PARIS).date().isoformat()


_FINISHED_MATCH_STATUSES = {"FT", "AET", "PEN"}
_VOID_MATCH_STATUSES = {"PST", "CANC", "ABD", "SUSP", "AWD"}


def evaluate_prediction(
    prediction_type: str,
    predicted_value: str,
    home_score,
    away_score,
    match_status: str = "FT",
) -> str:
    """
    Évalue un avis pré-match figé à partir du score final.

    Retourne uniquement les statuts métier historiques :
    pending, won, lost, void.
    """
    status = str(match_status or "NS").upper()
    if status in _VOID_MATCH_STATUSES:
        return "void"
    if status not in _FINISHED_MATCH_STATUSES:
        return "pending"
    if home_score is None or away_score is None:
        return "pending"

    try:
        home = int(home_score)
        away = int(away_score)
    except (TypeError, ValueError):
        return "pending"

    total_goals = home + away
    actual = "home" if home > away else "away" if away > home else "draw"

    type_key = str(prediction_type or "").strip().lower().replace(" ", "_")
    value_key = str(predicted_value or "").strip().upper().replace(",", ".")

    if type_key in {"over25", "over_25", "+2.5", "+2,5"} or value_key in {
        "OVER25",
        "OVER_25",
        "OVER 2.5",
        "O2.5",
        "+2.5",
        "2.5",
    }:
        return "won" if total_goals >= 3 else "lost"

    if type_key in {"over15", "over_15", "+1.5", "+1,5"} or value_key in {
        "OVER15",
        "OVER_15",
        "OVER 1.5",
        "O1.5",
        "+1.5",
        "1.5",
    }:
        return "won" if total_goals >= 2 else "lost"

    if type_key in {"btts", "l2m", "both_teams_score"} or value_key in {
        "BTTS",
        "L2M",
        "BTT",
        "BOTH TEAMS SCORE",
        "YES",
    }:
        return "won" if home > 0 and away > 0 else "lost"

    if value_key in {"HOME", "DOMICILE"}:
        value_key = "1"
    elif value_key in {"DRAW", "NUL", "X"}:
        value_key = "N"
    elif value_key in {"AWAY", "EXTERIEUR", "EXTÉRIEUR"}:
        value_key = "2"
    elif value_key == "1X":
        value_key = "1N"
    elif value_key == "X2":
        value_key = "N2"

    result_rules = {
        "1": {"home"},
        "N": {"draw"},
        "2": {"away"},
        "1N": {"home", "draw"},
        "N2": {"draw", "away"},
        "12": {"home", "away"},
    }
    if value_key in result_rules:
        return "won" if actual in result_rules[value_key] else "lost"

    return "pending"


# ══════════════════════════════════════════════
# CONNEXION SUPABASE
# ══════════════════════════════════════════════

def _get_client():
    """Retourne le client Supabase (singleton)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        log.warning("SUPABASE_URL ou SUPABASE_KEY non définis — mode hors-ligne")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        log.info("Supabase connecté (%s)", url)
        return _supabase_client
    except Exception as e:
        log.warning("Erreur connexion Supabase : %s", e)
        return None


# ══════════════════════════════════════════════
# CACHE DISQUE LOCAL
# ══════════════════════════════════════════════

def _load_local_cache() -> dict:
    """Charge le cache disque → {fixture_id: match_data}."""
    if _LOCAL_CACHE_FILE.exists():
        try:
            with open(_LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("Cache disque chargé : %d matchs", len(data))
            return data
        except Exception as e:
            log.warning("Cache disque corrompu : %s", e)
    return {}


def _save_local_cache(history: dict):
    """Sauvegarde atomique du cache disque."""
    _LOCAL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LOCAL_CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, default=str)
    tmp.replace(_LOCAL_CACHE_FILE)


def _save_json_cache(path: Path, data) -> None:
    """Sauvegarde atomique d'un fichier JSON cache générique."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        log.warning("Erreur sauvegarde cache %s : %s", path.name, e)


def _load_json_cache(path: Path):
    """Charge un fichier JSON cache générique, retourne None si absent/corrompu."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ══════════════════════════════════════════════
# HISTORICAL DATA
# ══════════════════════════════════════════════

def load_history(force_refresh: bool = False) -> dict:
    """
    Charge l'historique complet depuis Supabase + cache local.
    Delta sync : ne télécharge que les nouveaux fixture_ids.
    """
    local = {} if force_refresh else _load_local_cache()

    client = _get_client()
    if client is None:
        log.info("Supabase indisponible — cache local (%d matchs)", len(local))
        return local

    try:
        local_ids = set(local.keys())
        response = client.table("historical_data").select("fixture_id, full_data").execute()
        rows = response.data or []
        new_count = 0
        for row in rows:
            fid = str(row.get("fixture_id", ""))
            if not fid or fid in local_ids:
                continue
            if row.get("full_data"):
                local[fid] = row["full_data"]
                new_count += 1

        if new_count > 0:
            log.info("Delta sync Supabase : +%d matchs (total: %d)", new_count, len(local))
            _save_local_cache(local)
        else:
            log.info("Cache à jour : %d matchs", len(local))
    except Exception as e:
        log.warning("Erreur lecture Supabase historical_data : %s", e)

    return local


def append_matches(matches: dict, local_cache: Optional[dict] = None) -> int:
    """
    Ajoute des matchs à Supabase + cache local.
    Utilise UPSERT pour éviter les doublons.
    """
    if not matches:
        return 0

    if local_cache is None:
        local_cache = _load_local_cache()

    new_matches = {fid: data for fid, data in matches.items() if fid not in local_cache}
    if not new_matches:
        return 0

    rows = []
    for fid, data in new_matches.items():
        home = data.get("home_team", {})
        away = data.get("away_team", {})
        result = data.get("result", {})
        date_str = str(data.get("date", ""))[:10] or None
        rows.append({
            "fixture_id": str(data.get("fixture_id", fid)),
            "league_id": data.get("league_id"),
            "league_name": data.get("league_name", ""),
            "date": date_str,
            "home_team": home.get("name", "") if isinstance(home, dict) else str(home),
            "away_team": away.get("name", "") if isinstance(away, dict) else str(away),
            "home_goals": result.get("home_goals"),
            "away_goals": result.get("away_goals"),
            "full_data": data,
        })

    client = _get_client()
    written = 0

    if client is not None:
        try:
            for i in range(0, len(rows), 15):
                batch = rows[i:i + 15]
                client.table("historical_data").upsert(
                    batch, on_conflict="fixture_id"
                ).execute()
                written += len(batch)
            log.info("Supabase historical_data : +%d matchs écrits", written)
        except Exception as e:
            log.error("Erreur écriture Supabase historical_data : %s", e)
    else:
        written = len(rows)

    for fid, data in new_matches.items():
        local_cache[fid] = data
    _save_local_cache(local_cache)

    return written


def upsert_matches(matches: dict) -> int:
    """
    Étape 21.6 : UPSERT inconditionnel dans historical_data + local_cache.

    Contrairement à append_matches(), cette fonction NE filtre PAS contre
    local_cache : elle re-écrit le `full_data` même si le fixture existe
    déjà. Permet d'enrichir un fixture déjà connu avec :
      - odds_data (avec extended_markets + all_bookmakers_raw)
      - odds_fetched_at
      - prediction, _extra, features mis à jour

    Utilisée uniquement par le pipeline J/J+1 (api/routes/matches.py).
    Le cache local est mis à jour pour rester synchrone avec Supabase.

    Returns
    -------
    int : nombre de lignes upsertées en Supabase (+ local_cache mis à jour).
    """
    if not matches:
        return 0

    rows = []
    for fid, data in matches.items():
        if not data:
            continue
        home = data.get("home_team", {})
        away = data.get("away_team", {})
        result = data.get("result", {}) or {}
        # En J/J+1 le match peut être finished/live : on tente d'extraire le score
        hg = result.get("home_goals", data.get("current_home_goals"))
        ag = result.get("away_goals", data.get("current_away_goals"))
        date_str = str(data.get("date", ""))[:10] or None
        rows.append({
            "fixture_id":  str(data.get("fixture_id", fid)),
            "league_id":   data.get("league_id"),
            "league_name": data.get("league_name", ""),
            "date":        date_str,
            "home_team":   home.get("name", "") if isinstance(home, dict) else str(home),
            "away_team":   away.get("name", "") if isinstance(away, dict) else str(away),
            "home_goals":  hg,
            "away_goals":  ag,
            "full_data":   data,
        })

    if not rows:
        return 0

    client = _get_client()
    written = 0
    if client is not None:
        try:
            for i in range(0, len(rows), 15):
                batch = rows[i:i + 15]
                client.table("historical_data").upsert(
                    batch, on_conflict="fixture_id"
                ).execute()
                written += len(batch)
            log.info("Supabase historical_data UPSERT : %d matchs écrits/màj", written)
        except Exception as e:
            log.error("Erreur upsert historical_data Supabase : %s", e)
            return 0
    else:
        log.warning("Supabase indisponible — upsert_matches fallback local_cache uniquement")
        written = len(rows)

    # Maintenir local_cache à jour
    try:
        local_cache = _load_local_cache()
        for fid, data in matches.items():
            local_cache[str(fid)] = data
        _save_local_cache(local_cache)
    except Exception as e:
        log.warning("Sync local_cache après upsert_matches : %s", e)

    return written


# ══════════════════════════════════════════════
# BET HISTORY
# ══════════════════════════════════════════════

def save_bet_history(results: list, target_date: str = None) -> int:
    """
    Sauvegarde les prédictions du jour dans bet_history.
    Dédup par fixture_id (contrainte UNIQUE en base).
    """
    if not results:
        return 0

    if target_date is None:
        target_date = _paris_today_iso()

    client = _get_client()
    if client is None:
        log.warning("Supabase indisponible — bet_history non sauvegardé")
        return 0

    # Lire les fixture_ids existants — si lecture échoue, abort (évite doublons)
    try:
        resp = client.table("bet_history").select("fixture_id").execute()
        existing_fids = {str(r["fixture_id"]) for r in (resp.data or [])}
    except Exception as e:
        log.error("Erreur lecture bet_history Supabase : %s — abort", e)
        return 0

    # Étape 21.6 : split insert vs update.
    # Anciennement : on skippait silencieusement les fixtures déjà en base.
    # Maintenant : on insère les nouvelles ET on met à jour les colonnes
    # volatiles (odds + score live) sur les fixtures déjà connues.
    insert_rows = []
    update_rows = []  # (fid, update_dict)
    seen = set()
    for r in results:
        fid = str(r.get("fixture_id", ""))
        if not fid or fid in seen:
            continue
        seen.add(fid)

        pred = r.get("prediction", {})
        extra = r.get("_extra", {})
        smart = pred.get("smart_bet", {})
        home = r.get("home_team", {})
        away = r.get("away_team", {})

        # Étape 21.2 : extraction des odds pour persistance Supabase
        odds_data = r.get("odds_data") or {}
        if not isinstance(odds_data, dict):
            odds_data = {}
        extended = odds_data.get("extended_markets") or {}
        odd_over15_raw = r.get("odd_over15")
        odd_btts_raw   = r.get("odd_btts")

        def _f(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        row = {
            "date": str(r.get("date", target_date))[:10],
            "fixture_id": fid,
            "league_name": str(r.get("league_name", "")),
            "league_flag": str(r.get("league_flag", "")),
            "home_team": home.get("name", "") if isinstance(home, dict) else str(home),
            "away_team": away.get("name", "") if isinstance(away, dict) else str(away),
            "proba_over25": round(float(pred.get("proba_over25", 0)), 4),
            "prediction_label": str(pred.get("label", "")),
            "is_smart_bet": bool(smart.get("is_smart_bet", False)),
            "is_secure_bet": bool(extra.get("is_secure_bet", False)),
            "match_status": str(r.get("match_status", "NS")),
            "home_goals": r.get("current_home_goals"),
            "away_goals": r.get("current_away_goals"),
            "proba_o15": round(float(extra.get("proba_o15", 0)), 4),
            "proba_btts": round(float(extra.get("proba_btts", 0)), 4),
            "winner": str(extra.get("winner", "")),
            "winner_proba": round(float(extra.get("winner_proba", 0)), 4),
            "p_home_win": round(float(extra.get("p_home_win", 0)), 4),
            "p_draw": round(float(extra.get("p_draw", 0)), 4),
            "p_away_win": round(float(extra.get("p_away_win", 0)), 4),
            # ── Odds brutes par marché (étape 21.2) ──
            "odd_over25":    _f(extended.get("over_25")  or odds_data.get("avg_over_25")),
            "odd_under25":   _f(extended.get("under_25") or odds_data.get("avg_under_25")),
            "odd_over15":    _f(extended.get("over_15")  or odd_over15_raw),
            "odd_under15":   _f(extended.get("under_15")),
            "odd_btts_yes":  _f(extended.get("btts_yes") or odd_btts_raw),
            "odd_btts_no":   _f(extended.get("btts_no")),
            "odd_home":      _f(extended.get("home")),
            "odd_draw":      _f(extended.get("draw")),
            "odd_away":      _f(extended.get("away")),
            "odd_double_1n": _f(extended.get("double_1n")),
            "odd_double_n2": _f(extended.get("double_n2")),
            "odd_double_12": _f(extended.get("double_12")),
            "odds_bookmaker":    odds_data.get("bookmaker_name"),
            "odds_bookmaker_id": odds_data.get("bookmaker_id"),
            "odds_fetched_at":   r.get("odds_fetched_at") or odds_data.get("fetched_at"),
            "odds_full":         odds_data if odds_data else None,
        }
        if fid in existing_fids:
            # Étape 21.6 : update partiel — uniquement les champs volatiles
            # (odds + score live + match_status). On ne touche PAS aux prédictions
            # pré-match déjà figées en base (proba_over25, prediction_label, etc.).
            update_rows.append((fid, {
                "match_status":      row["match_status"],
                "home_goals":        row["home_goals"],
                "away_goals":        row["away_goals"],
                "odd_over25":        row["odd_over25"],
                "odd_under25":       row["odd_under25"],
                "odd_over15":        row["odd_over15"],
                "odd_under15":       row["odd_under15"],
                "odd_btts_yes":      row["odd_btts_yes"],
                "odd_btts_no":       row["odd_btts_no"],
                "odd_home":          row["odd_home"],
                "odd_draw":          row["odd_draw"],
                "odd_away":          row["odd_away"],
                "odd_double_1n":     row["odd_double_1n"],
                "odd_double_n2":     row["odd_double_n2"],
                "odd_double_12":     row["odd_double_12"],
                "odds_bookmaker":    row["odds_bookmaker"],
                "odds_bookmaker_id": row["odds_bookmaker_id"],
                "odds_fetched_at":   row["odds_fetched_at"],
                "odds_full":         row["odds_full"],
            }))
        else:
            insert_rows.append(row)

    if not insert_rows and not update_rows:
        return 0

    # Champs ajoutés étape 21 (présence conditionnelle à la migration SQL appliquée)
    _ODDS_COLS = (
        "odd_over25", "odd_under25", "odd_over15", "odd_under15",
        "odd_btts_yes", "odd_btts_no",
        "odd_home", "odd_draw", "odd_away",
        "odd_double_1n", "odd_double_n2", "odd_double_12",
        "odds_bookmaker", "odds_bookmaker_id", "odds_fetched_at", "odds_full",
    )

    def _strip_odds(batch):
        return [{k: v for k, v in row.items() if k not in _ODDS_COLS} for row in batch]

    n_inserted = 0
    n_updated  = 0

    # ── INSERTS ──
    if insert_rows:
        try:
            for i in range(0, len(insert_rows), 50):
                batch = insert_rows[i:i + 50]
                try:
                    client.table("bet_history").insert(batch).execute()
                    n_inserted += len(batch)
                except Exception as e_inner:
                    msg = str(e_inner)
                    if any(c in msg for c in _ODDS_COLS):
                        log.warning("bet_history insert sans colonnes odds_* "
                                      "(migration 21.1 pas appliquée ?) : %s — fallback", msg[:200])
                        client.table("bet_history").insert(_strip_odds(batch)).execute()
                        n_inserted += len(batch)
                    else:
                        raise
        except Exception as e:
            log.error("Erreur écriture bet_history INSERT Supabase : %s", e)

    # ── UPDATES (étape 21.6) ──
    if update_rows:
        try:
            for fid, upd in update_rows:
                try:
                    client.table("bet_history").update(upd).eq("fixture_id", fid).execute()
                    n_updated += 1
                except Exception as e_upd:
                    msg = str(e_upd)
                    if any(c in msg for c in _ODDS_COLS):
                        # Fallback sans colonnes odds (migration absente)
                        stripped = {k: v for k, v in upd.items() if k not in _ODDS_COLS}
                        if stripped:
                            client.table("bet_history").update(stripped).eq("fixture_id", fid).execute()
                            n_updated += 1
                    else:
                        log.warning("bet_history update fixture %s : %s", fid, msg[:200])
        except Exception as e:
            log.error("Erreur écriture bet_history UPDATE Supabase : %s", e)

    if n_inserted or n_updated:
        log.info("bet_history Supabase : +%d insert / %d updates pour %s",
                  n_inserted, n_updated, target_date)
    return n_inserted + n_updated


def load_bet_history(target_date: str = None) -> list:
    """Charge les prédictions d'une date depuis Supabase."""
    if target_date is None:
        target_date = _paris_today_iso()

    client = _get_client()
    if client is None:
        return []

    try:
        resp = (
            client.table("bet_history")
            .select("*")
            .eq("date", target_date[:10])
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        log.warning("Erreur lecture bet_history Supabase : %s", e)
        return []

    results = []
    seen_fids = set()
    for row in rows:
        fid = str(row.get("fixture_id", ""))
        if not fid or fid in seen_fids:
            continue
        seen_fids.add(fid)

        proba = float(row.get("proba_over25") or 0)
        results.append({
            "fixture_id": fid,
            "date": str(row.get("date", "")),
            "analysis_date": row.get("analysis_date"),
            "created_at": row.get("created_at"),
            "league_name": row.get("league_name", ""),
            "league_flag": row.get("league_flag", ""),
            "home_team": {"name": row.get("home_team", "")},
            "away_team": {"name": row.get("away_team", "")},
            "match_status": row.get("match_status", "NS"),
            "current_home_goals": row.get("home_goals"),
            "current_away_goals": row.get("away_goals"),
            "prediction": {
                "proba_over25": proba,
                "prediction": 1 if proba >= 0.55 else 0,
                "label": row.get("prediction_label", ""),
                "confidence": proba if proba >= 0.55 else 1 - proba,
                "smart_bet": {
                    "is_smart_bet": bool(row.get("is_smart_bet", False)),
                    "signals": [],
                },
            },
            "_extra": {
                "proba_o15": float(row.get("proba_o15") or 0),
                "proba_btts": float(row.get("proba_btts") or 0),
                "is_secure_bet": bool(row.get("is_secure_bet", False)),
                "winner": row.get("winner", ""),
                "winner_proba": float(row.get("winner_proba") or 0),
                "p_home_win": float(row.get("p_home_win") or 0),
                "p_draw": float(row.get("p_draw") or 0),
                "p_away_win": float(row.get("p_away_win") or 0),
            },
        })

    log.info("bet_history Supabase : %d prédictions pour %s", len(results), target_date)
    return results


def list_bet_history_dates() -> list:
    """Liste toutes les dates disponibles dans bet_history, tri décroissant."""
    from datetime import date as _d

    client = _get_client()
    if client is None:
        return _load_bet_history_dates_cache()

    try:
        resp = client.table("bet_history").select("date").execute()
        rows = resp.data or []
        dates_set = set()
        for row in rows:
            d_str = str(row.get("date", ""))[:10]
            if d_str:
                try:
                    dates_set.add(_d.fromisoformat(d_str))
                except ValueError:
                    pass
        result = sorted(dates_set, reverse=True)
        if result:
            _save_json_cache(
                _BET_HISTORY_DATES_CACHE,
                [d.isoformat() for d in result]
            )
        return result
    except Exception as e:
        log.warning("Erreur lecture dates Supabase : %s — fallback cache", e)
        return _load_bet_history_dates_cache()


def _load_bet_history_dates_cache() -> list:
    from datetime import date as _d
    raw = _load_json_cache(_BET_HISTORY_DATES_CACHE)
    if not raw:
        return []
    try:
        return sorted([_d.fromisoformat(d) for d in raw], reverse=True)
    except Exception:
        return []


def update_bet_history_scores(results: list, target_date: str = None) -> int:
    """Met à jour les scores finaux dans bet_history pour les matchs terminés."""
    if target_date is None:
        target_date = _paris_today_iso()

    evaluable = []
    for r in results:
        status = str(r.get("match_status", "NS")).upper()
        if status in _VOID_MATCH_STATUSES:
            evaluable.append(r)
            continue
        if (
            status in _FINISHED_MATCH_STATUSES
            and r.get("current_home_goals") is not None
            and r.get("current_away_goals") is not None
        ):
            evaluable.append(r)
    if not evaluable:
        return 0

    client = _get_client()
    if client is None:
        return 0

    updated = 0
    for r in evaluable:
        fid = str(r.get("fixture_id", ""))
        if not fid:
            continue
        match_status = str(r.get("match_status", "FT")).upper()
        try:
            update_payload = {"match_status": match_status}
            if match_status in _FINISHED_MATCH_STATUSES:
                home_goals = int(r["current_home_goals"])
                away_goals = int(r["current_away_goals"])
                update_payload.update({
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                })

                extra = r.get("_extra") or {}
                result_pick = {
                    "home": "1",
                    "draw": "N",
                    "away": "2",
                }.get(str(extra.get("winner", "")).lower(), "")
                evaluations = {
                    "over25": evaluate_prediction("over25", "+2.5", home_goals, away_goals, match_status),
                    "over15": evaluate_prediction("over15", "+1.5", home_goals, away_goals, match_status),
                    "btts": evaluate_prediction("btts", "L2M", home_goals, away_goals, match_status),
                    "result": evaluate_prediction("result", result_pick, home_goals, away_goals, match_status),
                }
                log.debug("Évaluation fixture %s : %s", fid, evaluations)

            client.table("bet_history").update(update_payload).eq("fixture_id", fid).execute()
            updated += 1
        except Exception as e:
            log.warning("Erreur update score fixture %s : %s", fid, e)

    if updated > 0:
        log.info("bet_history Supabase : %d scores mis à jour (%s)", updated, target_date)
    return updated


def cleanup_bet_history_duplicates() -> int:
    """
    Avec Supabase, la contrainte UNIQUE(fixture_id) empêche les doublons à l'insertion.
    Fonction conservée pour compatibilité avec app.py.
    """
    log.info("cleanup_bet_history_duplicates : contrainte UNIQUE Supabase active, rien à faire")
    return 0
