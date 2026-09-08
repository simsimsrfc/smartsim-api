"""
Smart Sim — Data Fetcher
Extraction complète via API-FOOTBALL PRO (v3)
Cache disque + gestion de quota + batching
"""

import time
import json
import hashlib
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import diskcache

from config import (
    BASE_URL, HEADERS, LEAGUES, DAILY_QUOTA, BATCH_SIZE,
    CACHE_DIR, CACHE_TTL_FIXTURES, CACHE_TTL_STATS, CACHE_TTL_ODDS, CACHE_TTL_REFEREE,
    LAST_N_MATCHES, H2H_LIMIT, BOOKMAKERS, BOOKMAKERS_MIN, STATS_KEYS,
    FETCH_LINEUPS, CACHE_TTL_LINEUPS,
    EUROPEAN_CUP_IDS, EURO_CUP_HISTORY_SEASONS,
)

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SmartSim.fetcher")

# Raisons d'absence à exclure du comptage (joueur disponible malgré l'entrée)
_EXCLUDE_INJURY_REASONS = frozenset({
    "international duty", "international", "national duty",
    "called up", "trêve internationale",
})
_VALID_INJURY_TYPES = frozenset({"injury", "suspension"})

# ──────────────────────────────────────────────
# CACHE (diskcache — persistant sur disque)
# ──────────────────────────────────────────────
_cache = diskcache.Cache(CACHE_DIR, size_limit=500 * 1024 * 1024)  # 500 MB max


# ──────────────────────────────────────────────
# QUOTA TRACKER
# ──────────────────────────────────────────────
class QuotaTracker:
    """Suit le nombre de requêtes API effectuées aujourd'hui."""

    def __init__(self):
        self._key = f"quota_{date.today().isoformat()}"

    @property
    def used(self) -> int:
        return _cache.get(self._key, 0)

    @property
    def remaining(self) -> int:
        return max(0, DAILY_QUOTA - self.used)

    def increment(self):
        current = self.used
        _cache.set(self._key, current + 1, expire=86400)

    def can_call(self, n: int = 1) -> bool:
        return self.remaining >= n


quota = QuotaTracker()


# ──────────────────────────────────────────────
# CORE API CALLER (avec cache + retry + quota)
# ──────────────────────────────────────────────
def _cache_key(endpoint: str, params: dict) -> str:
    """Génère une clé de cache déterministe."""
    raw = f"{endpoint}|{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class QuotaExceeded(Exception):
    """Levée quand le quota API est épuisé ou un 429 persistant."""
    pass


# Pause entre chaque appel API (anti-ban : 1.2s = ~50 req/min, bien sous les 300)
API_SLEEP = 1.2


def api_get(endpoint: str, params: dict, ttl: int = CACHE_TTL_STATS,
            retries: int = 3) -> Optional[dict]:
    """
    Appel GET avec :
    - Cache disque (clé = hash endpoint+params)
    - Quota check (lève QuotaExceeded si épuisé)
    - Retry avec backoff exponentiel
    - Pause anti-ban de 1.2s entre chaque appel
    """
    key = _cache_key(endpoint, params)

    # 1. Cache hit ?
    cached = _cache.get(key)
    if cached is not None:
        return cached

    # 2. Quota check
    if not quota.can_call():
        log.warning("⛔ Quota journalier atteint (%d/%d). Arrêt des appels.", quota.used, DAILY_QUOTA)
        raise QuotaExceeded(f"Quota épuisé : {quota.used}/{DAILY_QUOTA}")

    # 3. Appel API
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            quota.increment()

            if resp.status_code == 429:
                wait = 5 * attempt  # Backoff plus agressif : 5s, 10s, 15s
                log.warning("⚠️ Rate-limited (429). Pause de %ds…", wait)
                time.sleep(wait)
                if attempt == retries:
                    raise QuotaExceeded("Rate-limit 429 persistant après 3 tentatives")
                continue

            resp.raise_for_status()
            data = resp.json()

            # API-Football renvoie errors[] en cas de souci
            errors = data.get("errors")
            if errors:
                err_str = str(errors).lower()
                if "quota" in err_str or "rate" in err_str or "limit" in err_str:
                    log.warning("⛔ Quota/Rate limit détecté dans la réponse API.")
                    raise QuotaExceeded(f"API quota error: {errors}")
                log.error("API error pour %s %s : %s", endpoint, params, errors)
                return None

            # Mise en cache
            _cache.set(key, data, expire=ttl)

            # Anti-ban : pause longue entre chaque appel
            time.sleep(API_SLEEP)
            return data

        except QuotaExceeded:
            raise  # Propager immédiatement
        except requests.exceptions.Timeout:
            log.warning("Timeout sur %s (tentative %d/%d)", endpoint, attempt, retries)
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            log.error("Erreur HTTP sur %s : %s", endpoint, e)
            if attempt == retries:
                return None
            time.sleep(3 * attempt)

    return None


# ══════════════════════════════════════════════
# 1. FIXTURES DU JOUR
# ══════════════════════════════════════════════
def fetch_fixtures_today(league_id: int, season: int) -> list[dict]:
    """
    Récupère TOUS les matchs du jour pour une ligue (NS, LIVE, FT, AET, PEN…).
    Aucun filtre de statut — on veut la journée complète de 00:00 à 23:59.
    """
    today = date.today().isoformat()
    data = api_get("fixtures", {
        "league": league_id,
        "season": season,
        "date": today,
    }, ttl=CACHE_TTL_FIXTURES)

    if not data:
        return []
    return data.get("response", [])


def fetch_all_fixtures_today() -> dict[int, list[dict]]:
    """
    Récupère les fixtures du jour pour TOUTES les ligues configurées.
    Retourne {league_id: [fixtures]}.
    """
    all_fixtures = {}
    for league_id, meta in LEAGUES.items():
        if not quota.can_call():
            log.warning("Quota épuisé, arrêt du fetch fixtures.")
            break
        fixtures = fetch_fixtures_today(league_id, meta["season"])
        if fixtures:
            all_fixtures[league_id] = fixtures
            log.info("  %s %s : %d match(s)", meta["flag"], meta["name"], len(fixtures))
    return all_fixtures


# ══════════════════════════════════════════════
# 2. STATISTIQUES DE MATCH (match terminé)
# ══════════════════════════════════════════════
def fetch_fixture_stats(fixture_id: int) -> Optional[dict]:
    """
    Statistiques détaillées d'un match terminé.
    Retourne {home: {stat: val}, away: {stat: val}}.
    """
    data = api_get("fixtures/statistics", {
        "fixture": fixture_id,
    }, ttl=CACHE_TTL_STATS)

    if not data or not data.get("response"):
        return None

    result = {}
    for team_stats in data["response"]:
        team_name = team_stats.get("team", {}).get("name", "Unknown")
        stats = {}
        for s in team_stats.get("statistics", []):
            stat_type = s.get("type", "")
            stat_value = s.get("value")
            # Nettoyer les pourcentages ("65%" → 65.0)
            if isinstance(stat_value, str) and stat_value.endswith("%"):
                try:
                    stat_value = float(stat_value.strip("%"))
                except ValueError:
                    pass
            stats[stat_type] = stat_value
        result[team_name] = stats

    return result


# ══════════════════════════════════════════════
# 3. DERNIERS MATCHS D'UNE ÉQUIPE (form)
# ══════════════════════════════════════════════
def fetch_last_matches(team_id: int, n: int = LAST_N_MATCHES) -> list[dict]:
    """Récupère les n derniers matchs terminés d'une équipe."""
    data = api_get("fixtures", {
        "team": team_id,
        "last": n,
    }, ttl=CACHE_TTL_STATS)

    if not data:
        return []
    return data.get("response", [])


def fetch_last_matches_with_stats(team_id: int, n: int = LAST_N_MATCHES) -> list[dict]:
    """
    Récupère les n derniers matchs + les stats de chaque match.
    Retourne une liste enrichie de fixtures avec clé 'match_stats'.
    """
    matches = fetch_last_matches(team_id, n)
    enriched = []
    for match in matches:
        fixture_id = match.get("fixture", {}).get("id")
        match_data = dict(match)
        if fixture_id:
            match_data["match_stats"] = fetch_fixture_stats(fixture_id)
        enriched.append(match_data)
    return enriched


def fetch_team_competition_history(team_id: int, league_id: int,
                                   n_seasons: int = EURO_CUP_HISTORY_SEASONS) -> list[dict]:
    """
    Récupère l'historique d'une équipe dans une compétition spécifique
    sur les N dernières saisons. Pour les coupes européennes, cela donne
    l'expérience et le niveau de performance dans cette coupe précise.
    Retourne une liste de fixtures enrichies (match_stats + goal_timings).
    """
    current_season = LEAGUES.get(league_id, {}).get("season", 2025)
    all_matches = []

    for offset in range(n_seasons):
        season = current_season - offset
        data = api_get("fixtures", {
            "team": team_id,
            "league": league_id,
            "season": season,
            "status": "FT-AET-PEN",
        }, ttl=CACHE_TTL_STATS)

        if data and data.get("response"):
            all_matches.extend(data["response"])

    if not all_matches:
        return []

    # Trier par date desc et enrichir les stats
    all_matches.sort(
        key=lambda m: m.get("fixture", {}).get("date", ""), reverse=True
    )
    enriched = []
    for match in all_matches[:20]:  # Cap à 20 matchs max
        fixture_id = match.get("fixture", {}).get("id")
        m = dict(match)
        if fixture_id:
            m["match_stats"] = fetch_fixture_stats(fixture_id)
        enriched.append(m)
    return enriched


# ══════════════════════════════════════════════
# 4. HEAD-TO-HEAD (H2H)
# ══════════════════════════════════════════════
def fetch_h2h(team_id_home: int, team_id_away: int, n: int = H2H_LIMIT) -> list[dict]:
    """Confrontations directes entre deux équipes."""
    data = api_get("fixtures/headtohead", {
        "h2h": f"{team_id_home}-{team_id_away}",
        "last": n,
    }, ttl=CACHE_TTL_STATS)

    if not data:
        return []
    return data.get("response", [])


def fetch_h2h_with_stats(team_id_home: int, team_id_away: int,
                          n: int = H2H_LIMIT) -> list[dict]:
    """H2H enrichi avec les stats de chaque confrontation."""
    matches = fetch_h2h(team_id_home, team_id_away, n)
    enriched = []
    for match in matches:
        fixture_id = match.get("fixture", {}).get("id")
        match_data = dict(match)
        if fixture_id:
            match_data["match_stats"] = fetch_fixture_stats(fixture_id)
        enriched.append(match_data)
    return enriched


# ══════════════════════════════════════════════
# 5. ÉVÉNEMENTS DE MATCH (buts, timing)
# ══════════════════════════════════════════════
def fetch_fixture_events(fixture_id: int) -> list[dict]:
    """Événements d'un match (buts, cartons, subs…)."""
    data = api_get("fixtures/events", {
        "fixture": fixture_id,
    }, ttl=CACHE_TTL_STATS)

    if not data:
        return []
    return data.get("response", [])


def extract_goal_timings(events: list[dict]) -> dict:
    """
    Extrait les timings de buts depuis les événements.
    Retourne :
      - first_goal_minute : minute du 1er but (None si 0-0)
      - goals_last_15 : nombre de buts après la 75e minute
      - goal_minutes : liste de toutes les minutes de but
    """
    goal_minutes = []
    for ev in events:
        if ev.get("type") == "Goal" and ev.get("detail") != "Missed Penalty":
            elapsed = ev.get("time", {}).get("elapsed")
            if elapsed is not None:
                goal_minutes.append(int(elapsed))

    goal_minutes.sort()
    return {
        "first_goal_minute": goal_minutes[0] if goal_minutes else None,
        "goals_last_15": sum(1 for m in goal_minutes if m >= 76),
        "goal_minutes": goal_minutes,
        "total_goals": len(goal_minutes),
    }


def fetch_goal_timings_for_matches(matches: list[dict]) -> list[dict]:
    """Enrichit une liste de matchs avec les timings de buts."""
    enriched = []
    for match in matches:
        fixture_id = match.get("fixture", {}).get("id")
        match_data = dict(match)
        if fixture_id:
            events = fetch_fixture_events(fixture_id)
            match_data["goal_timings"] = extract_goal_timings(events)
        enriched.append(match_data)
    return enriched


# ══════════════════════════════════════════════
# 6. ARBITRE (referee stats & tendance)
# ══════════════════════════════════════════════
def extract_referee_name(fixture: dict) -> Optional[str]:
    """Extrait le nom de l'arbitre depuis une fixture."""
    return fixture.get("fixture", {}).get("referee")


def fetch_referee_stats(referee_name: str, season: int) -> Optional[dict]:
    """
    Cherche les stats de l'arbitre via ses matchs récents.
    Calcule :
    - avg_fouls, avg_yellows, avg_reds
    - penalty_rate (penaltys / match)
    - avg_total_goals (tendance Over de l'arbitre)
    """
    if not referee_name:
        return None

    # Nettoyer le nom (API renvoie parfois "P. LastName, Country")
    clean_name = referee_name.split(",")[0].strip()

    # Récupérer les matchs terminés de la saison, puis filtrer par arbitre localement
    # (le paramètre "referee" n'est plus accepté par l'API)
    data = api_get("fixtures", {
        "season": season,
        "last": 50,
    }, ttl=CACHE_TTL_REFEREE)

    if not data or not data.get("response"):
        return None

    # Filtrer localement sur le nom de l'arbitre
    matches = [
        m for m in data["response"]
        if clean_name.lower() in (m.get("fixture", {}).get("referee") or "").lower()
    ]

    total = len(matches)
    if total == 0:
        return None

    total_fouls = 0
    total_yellows = 0
    total_reds = 0
    total_goals = 0
    total_penalties = 0

    for match in matches:
        fixture_id = match.get("fixture", {}).get("id")
        # Buts
        home_goals = match.get("goals", {}).get("home") or 0
        away_goals = match.get("goals", {}).get("away") or 0
        total_goals += home_goals + away_goals

        # Événements pour penaltys, cartons
        try:
            events = fetch_fixture_events(fixture_id)
            for ev in events:
                ev_type = ev.get("type", "")
                ev_detail = ev.get("detail", "")
                if ev_type == "Card":
                    if "Yellow" in ev_detail:
                        total_yellows += 1
                    elif "Red" in ev_detail:
                        total_reds += 1
                if ev_type == "Goal" and ev_detail == "Penalty":
                    total_penalties += 1
                if ev_type == "Goal" and ev_detail == "Missed Penalty":
                    total_penalties += 1
        except Exception as e:
            log.warning("Erreur events arbitre fixture %s : %s", fixture_id, e)

    # Estimation des fautes via les stats
    total_fouls = 0
    for match in matches[:5]:
        fixture_id = match.get("fixture", {}).get("id")
        try:
            stats = fetch_fixture_stats(fixture_id)
            if stats:
                for team_name, team_stats in stats.items():
                    fouls = team_stats.get("Fouls")
                    if fouls is not None:
                        total_fouls += int(fouls)
        except Exception:
            pass

    return {
        "referee_name": clean_name,
        "matches_analyzed": total,
        "avg_total_goals": round(total_goals / total, 2),
        "avg_yellows_per_match": round(total_yellows / total, 2),
        "avg_reds_per_match": round(total_reds / total, 2),
        "penalty_rate": round(total_penalties / total, 2),
        "avg_fouls_per_match": round(total_fouls / min(5, total), 2) if total_fouls else None,
        "over_25_rate": round(
            sum(1 for m in matches
                if (m.get("goals", {}).get("home") or 0) + (m.get("goals", {}).get("away") or 0) > 2)
            / total, 2
        ),
    }


# ══════════════════════════════════════════════
# 7. COTES (ODDS) — Over/Under 2.5 + Over/Under 1.5 + BTTS
# ══════════════════════════════════════════════
def _fetch_odds_raw(fixture_id: int) -> Optional[dict]:
    """
    UN SEUL appel API pour récupérer toutes les cotes d'un match.

    Note : on N'IMPOSE PLUS bookmaker=8 (Bet365). L'API renvoie tous les
    bookmakers disponibles en 1 appel, ce qui permet d'agréger un consensus
    plus robuste et de couvrir les ligues mineures où Bet365 est absent.
    Bet365 reste prioritaire en cas de présence (cf. _parse_all_odds).
    """
    data = api_get("odds", {
        "fixture": fixture_id,
    }, ttl=CACHE_TTL_ODDS)
    return data


def _parse_all_odds(data: Optional[dict]) -> dict:
    """
    Parse la réponse brute /odds et extrait tous les marchés utiles
    avec une priorité bookmaker (Bet365 prioritaire, fallback sur autres).

    Bet IDs API-Football v3 :
      - id 1  = Match Winner (Home / Draw / Away)
      - id 5  = Goals Over/Under (Over 1.5, Under 1.5, Over 2.5, Under 2.5…)
      - id 6  = Both Teams Score (Yes / No)
      - id 12 = Double Chance (Home/Draw, Home/Away, Draw/Away)

    Retour rétro-compatible (mêmes clés que l'ancienne version) +
    nouvelles clés étendues : home/draw/away, double_chance_*, all_bookmakers,
    market_count, bookmaker_count.
    """
    result = {
        "over_25": None, "under_25": None,
        "over_15": None, "under_15": None,
        "btts_yes": None, "btts_no": None,
        "home":      None, "draw":      None, "away":      None,
        "double_1n": None, "double_12": None, "double_n2": None,
        "bookmaker":     None,
        "bookmaker_id":  None,
        "all_bookmakers": [],
        "bookmaker_count": 0,
        "market_count":    0,
    }
    if not data or not data.get("response"):
        return result

    PREFERRED_BK = 8  # Bet365

    # Pass 1 : collecter par bookmaker
    by_bk = []  # liste de dict {id, name, markets_dict}
    for entry in data["response"]:
        for bk in entry.get("bookmakers", []):
            bk_id   = bk.get("id")
            bk_name = bk.get("name", "")
            mk = {"over_25": None, "under_25": None, "over_15": None, "under_15": None,
                  "btts_yes": None, "btts_no": None,
                  "home": None, "draw": None, "away": None,
                  "double_1n": None, "double_12": None, "double_n2": None}
            for bet in bk.get("bets", []):
                bet_id = bet.get("id")
                if bet_id == 1:
                    for val in bet.get("values", []):
                        v = str(val.get("value", ""))
                        odd = _safe_float(val.get("odd"))
                        if v == "Home":   mk["home"] = odd
                        elif v == "Draw": mk["draw"] = odd
                        elif v == "Away": mk["away"] = odd
                elif bet_id == 5:
                    for val in bet.get("values", []):
                        v = str(val.get("value", ""))
                        odd = _safe_float(val.get("odd"))
                        if v == "Over 2.5":    mk["over_25"]  = odd
                        elif v == "Under 2.5": mk["under_25"] = odd
                        elif v == "Over 1.5":  mk["over_15"]  = odd
                        elif v == "Under 1.5": mk["under_15"] = odd
                elif bet_id == 6:
                    for val in bet.get("values", []):
                        v = str(val.get("value", "")).lower()
                        odd = _safe_float(val.get("odd"))
                        if v == "yes": mk["btts_yes"] = odd
                        elif v == "no": mk["btts_no"] = odd
                elif bet_id == 12:
                    for val in bet.get("values", []):
                        v = str(val.get("value", ""))
                        odd = _safe_float(val.get("odd"))
                        if v == "Home/Draw":   mk["double_1n"] = odd
                        elif v == "Home/Away": mk["double_12"] = odd
                        elif v == "Draw/Away": mk["double_n2"] = odd
            if any(v is not None for v in mk.values()):
                by_bk.append({"id": bk_id, "name": bk_name, "markets": mk})

    if not by_bk:
        return result

    # Pass 2 : priorité Bet365, sinon premier bookmaker qui a la donnée par marché.
    def _pick(field):
        # 1. Cherche dans Bet365 (id=8)
        for b in by_bk:
            if b["id"] == PREFERRED_BK and b["markets"].get(field) is not None:
                return b["markets"][field], b["id"], b["name"]
        # 2. Sinon premier non-null
        for b in by_bk:
            v = b["markets"].get(field)
            if v is not None:
                return v, b["id"], b["name"]
        return None, None, None

    chosen_bk_id = None
    chosen_bk_name = None
    for field in ("over_25", "under_25", "over_15", "under_15",
                  "btts_yes", "btts_no", "home", "draw", "away",
                  "double_1n", "double_12", "double_n2"):
        v, bk_id, bk_name = _pick(field)
        result[field] = v
        if v is not None and chosen_bk_id is None:
            chosen_bk_id, chosen_bk_name = bk_id, bk_name

    result["bookmaker"]    = chosen_bk_name
    result["bookmaker_id"] = chosen_bk_id
    result["bookmaker_count"] = len(by_bk)
    result["all_bookmakers"]  = by_bk
    result["market_count"] = sum(1 for k in ("over_25", "over_15", "btts_yes",
                                              "home", "double_1n") if result[k] is not None)
    return result


def fetch_odds_over25(fixture_id: int) -> list[dict]:
    """
    Récupère les cotes Over 2.5 / Under 2.5 pour un match.
    Retourne une liste de {bookmaker, over_25, under_25} pour compatibilité
    avec get_market_consensus().

    Multi-bookmakers : un entry par bookmaker présent dans la réponse API
    (au lieu d'un seul Bet365). is_preferred = True si bookmaker_id ∈ BOOKMAKERS.
    """
    raw = _fetch_odds_raw(fixture_id)
    parsed = _parse_all_odds(raw)

    if parsed["over_25"] is None and not parsed["all_bookmakers"]:
        return []

    bookmaker_ids = set(BOOKMAKERS.keys())
    result = []
    for bk in parsed["all_bookmakers"]:
        mk = bk.get("markets", {})
        if mk.get("over_25") is None:
            continue
        result.append({
            "bookmaker_id": bk.get("id"),
            "bookmaker":    bk.get("name", ""),
            "over_25":      mk.get("over_25"),
            "under_25":     mk.get("under_25"),
            "is_preferred": bk.get("id") in bookmaker_ids,
        })
    return result


def fetch_odds_over15(fixture_id: int) -> Optional[float]:
    """Récupère la cote Over 1.5 (via cache du même appel API)."""
    raw = _fetch_odds_raw(fixture_id)
    parsed = _parse_all_odds(raw)
    return parsed["over_15"]


def fetch_odds_btts(fixture_id: int) -> Optional[float]:
    """Récupère la cote BTTS Yes (via cache du même appel API)."""
    raw = _fetch_odds_raw(fixture_id)
    parsed = _parse_all_odds(raw)
    return parsed["btts_yes"]


def get_market_consensus(odds_list: list[dict]) -> Optional[dict]:
    """
    Calcule le consensus marché à partir des cotes Over 2.5.
    Retourne :
    - avg_over_25 : cote moyenne Over 2.5
    - implied_prob : probabilité implicite moyenne
    - market_direction : 'over' ou 'under'
    - spread : écart max entre bookmakers (détection mouvement)
    """
    if not odds_list:
        return None

    # Prendre les bookmakers préférés d'abord, sinon tous
    preferred = [o for o in odds_list if o["is_preferred"]]
    sample = preferred[:BOOKMAKERS_MIN] if len(preferred) >= BOOKMAKERS_MIN else odds_list[:BOOKMAKERS_MIN]

    if not sample:
        return None

    over_odds = [o["over_25"] for o in sample if o["over_25"]]
    under_odds = [o["under_25"] for o in sample if o["under_25"]]

    if not over_odds:
        return None

    avg_over = sum(over_odds) / len(over_odds)
    avg_under = sum(under_odds) / len(under_odds) if under_odds else None

    # Probabilité implicite (1/cote, normalisée)
    implied_over = 1 / avg_over if avg_over > 0 else 0
    implied_under = 1 / avg_under if avg_under and avg_under > 0 else 0
    total_implied = implied_over + implied_under
    if total_implied > 0:
        implied_over_norm = implied_over / total_implied
    else:
        implied_over_norm = implied_over

    # Spread (volatilité du marché)
    spread = max(over_odds) - min(over_odds) if len(over_odds) > 1 else 0

    return {
        "avg_over_25": round(avg_over, 3),
        "avg_under_25": round(avg_under, 3) if avg_under else None,
        "implied_prob_over": round(implied_over_norm, 4),
        "market_direction": "over" if implied_over_norm > 0.5 else "under",
        "spread": round(spread, 3),
        "bookmakers_count": len(sample),
        "all_odds": odds_list,
        # Étape 21.4 : timestamp anti-leakage. UTC ISO 8601.
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════
# 8. DONNÉES ÉQUIPE (classement, forme)
# ══════════════════════════════════════════════
def fetch_team_standings(league_id: int, season: int) -> dict[int, dict]:
    """
    Classement d'une ligue.
    Retourne {team_id: {rank, points, form, home, away, goals_for, goals_against}}.
    """
    data = api_get("standings", {
        "league": league_id,
        "season": season,
    }, ttl=CACHE_TTL_STATS)

    if not data or not data.get("response"):
        return {}

    standings = {}
    for league_data in data["response"]:
        for group in league_data.get("league", {}).get("standings", []):
            for team in group:
                tid = team.get("team", {}).get("id")
                if tid:
                    standings[tid] = {
                        "rank": team.get("rank"),
                        "points": team.get("points"),
                        "form": team.get("form", ""),
                        "played": team.get("all", {}).get("played", 0),
                        "goals_for": team.get("all", {}).get("goals", {}).get("for", 0),
                        "goals_against": team.get("all", {}).get("goals", {}).get("against", 0),
                        "home_wins": team.get("home", {}).get("win", 0),
                        "home_draws": team.get("home", {}).get("draw", 0),
                        "home_losses": team.get("home", {}).get("lose", 0),
                        "home_goals_for": team.get("home", {}).get("goals", {}).get("for", 0),
                        "home_goals_against": team.get("home", {}).get("goals", {}).get("against", 0),
                        "away_wins": team.get("away", {}).get("win", 0),
                        "away_draws": team.get("away", {}).get("draw", 0),
                        "away_losses": team.get("away", {}).get("lose", 0),
                        "away_goals_for": team.get("away", {}).get("goals", {}).get("for", 0),
                        "away_goals_against": team.get("away", {}).get("goals", {}).get("against", 0),
                    }
    return standings


# ══════════════════════════════════════════════
# 9. LINEUPS & INJURIES (compositions + absences)
# ══════════════════════════════════════════════
def fetch_lineups(fixture_id: int) -> Optional[dict]:
    """
    Récupère les compositions d'équipes (titulaires + remplaçants).
    Retourne {home: {formation, startXI, substitutes}, away: ...}.
    """
    if not FETCH_LINEUPS:
        return None

    data = api_get("fixtures/lineups", {
        "fixture": fixture_id,
    }, ttl=CACHE_TTL_LINEUPS)

    if not data or not data.get("response"):
        return None

    result = {}
    for team_data in data["response"]:
        team_name = team_data.get("team", {}).get("name", "Unknown")
        result[team_name] = {
            "formation": team_data.get("formation"),
            "startXI": [
                p.get("player", {}).get("name")
                for p in team_data.get("startXI", [])
            ],
            "substitutes": [
                p.get("player", {}).get("name")
                for p in team_data.get("substitutes", [])
            ],
        }
    return result


def fetch_injuries(fixture_id: int) -> list[dict]:
    """
    Récupère les joueurs blessés / suspendus pour un match.
    Retourne une liste de {team, player, type, reason}.
    """
    if not FETCH_LINEUPS:
        return []

    data = api_get("injuries", {
        "fixture": fixture_id,
    }, ttl=CACHE_TTL_LINEUPS)

    if not data or not data.get("response"):
        return []

    injuries = []
    for entry in data["response"]:
        injuries.append({
            "team": entry.get("team", {}).get("name"),
            "team_id": entry.get("team", {}).get("id"),
            "player": entry.get("player", {}).get("name"),
            "type": entry.get("player", {}).get("type"),
            "reason": entry.get("player", {}).get("reason"),
        })
    return injuries


def count_missing_players(injuries: list[dict], team_id: int) -> int:
    """
    Compte le nombre de joueurs réellement indisponibles pour le match.
    Exclut les convocations internationales (joueur DISPONIBLE après trêve).
    Ne compte que les blessures et suspensions actives.
    """
    count = 0
    for inj in injuries:
        if inj.get("team_id") != team_id:
            continue
        reason = (inj.get("reason") or "").strip().lower()
        inj_type = (inj.get("type") or "").strip().lower()
        if any(excl in reason for excl in _EXCLUDE_INJURY_REASONS):
            continue
        if inj_type in _VALID_INJURY_TYPES or reason:
            count += 1
    return count


def weighted_missing_score(injuries: list[dict], team_id: int,
                           team_lineups: Optional[dict]) -> float:
    """
    Pondère les blessés selon leur statut dans les lineups du match :
    - Joueur dans startXI → 0.0 (il joue, faux positif)
    - Joueur dans substitutes → 0.3 (backup, impact réduit)
    - Joueur absent des deux → 1.0 (vraie absence)
    Fallback sur le compte simple si lineups absent.
    """
    if not team_lineups:
        return float(count_missing_players(injuries, team_id))

    startxi = set(team_lineups.get("startXI", []) or [])
    subs = set(team_lineups.get("substitutes", []) or [])
    score = 0.0
    for inj in injuries:
        if inj.get("team_id") != team_id:
            continue
        reason = (inj.get("reason") or "").strip().lower()
        if any(excl in reason for excl in _EXCLUDE_INJURY_REASONS):
            continue
        player = (inj.get("player") or "").strip()
        if not player:
            continue
        if player in startxi:
            continue  # Il joue → faux positif
        if player in subs:
            score += 0.3
        else:
            score += 1.0
    return round(score, 2)


def formation_offensive_score(formation: Optional[str]) -> int:
    """
    Score offensif d'une formation : 1 = offensive, 0 = neutre, -1 = défensive.
    4-3-3, 3-4-3, 4-2-3-1 → offensif | 5-3-2, 5-4-1 → défensif | 4-4-2 → neutre.
    """
    if not formation:
        return 0
    parts = formation.replace(" ", "").split("-")
    if len(parts) < 3:
        return 0
    try:
        defenders = int(parts[0])
        attackers = int(parts[-1])
    except ValueError:
        return 0
    if defenders >= 5:
        return -1
    if attackers >= 3:
        return 1
    return 0


# ══════════════════════════════════════════════
# 10. PIPELINE COMPLET POUR UN MATCH
# ══════════════════════════════════════════════
def fetch_full_match_data(fixture: dict, league_id: int, season: int) -> Optional[dict]:
    """
    Pipeline complet d'extraction pour UN match (à venir, en cours ou terminé).
    Rassemble toutes les données nécessaires au feature engineering.
    Inclut lineups + injuries si FETCH_LINEUPS est activé.
    Propage le statut du match (NS, LIVE, FT…) et le score actuel.
    """
    fixture_info = fixture.get("fixture", {})
    fixture_id = fixture_info.get("id")
    home_team = fixture.get("teams", {}).get("home", {})
    away_team = fixture.get("teams", {}).get("away", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")

    if not all([fixture_id, home_id, away_id]):
        log.warning("Fixture incomplète : %s", fixture_id)
        return None

    # ── Statut du match (NS, 1H, 2H, HT, FT, AET, PEN, LIVE…) ──
    status_data = fixture_info.get("status", {})
    match_status = status_data.get("short", "NS")   # NS par défaut
    match_elapsed = status_data.get("elapsed")       # minute en cours (None si NS/FT)

    # ── Score actuel (live ou final) ──
    goals_data = fixture.get("goals", {})
    current_home_goals = goals_data.get("home")      # None si NS
    current_away_goals = goals_data.get("away")

    log.info("── Fetch complet : %s vs %s (fixture %d, status=%s) ──",
             home_team.get("name"), away_team.get("name"), fixture_id, match_status)

    # ① Derniers matchs de chaque équipe + stats
    home_last = fetch_last_matches_with_stats(home_id, LAST_N_MATCHES)
    away_last = fetch_last_matches_with_stats(away_id, LAST_N_MATCHES)

    # ② Goal timings pour les derniers matchs
    home_last = fetch_goal_timings_for_matches(home_last)
    away_last = fetch_goal_timings_for_matches(away_last)

    # ③ H2H + stats
    h2h = fetch_h2h_with_stats(home_id, away_id, H2H_LIMIT)
    h2h = fetch_goal_timings_for_matches(h2h)

    # ④ Arbitre
    referee_name = extract_referee_name(fixture)
    referee_data = fetch_referee_stats(referee_name, season) if referee_name else None

    # ⑤ Cotes — multi-bookmakers (Bet365 prioritaire en fallback)
    # Marchés : 1X2, Over/Under 2.5, Over/Under 1.5, BTTS, Double Chance
    odds = fetch_odds_over25(fixture_id)
    market = get_market_consensus(odds)
    odd_over15 = fetch_odds_over15(fixture_id)
    odd_btts = fetch_odds_btts(fixture_id)
    # Parse complet (1 seul appel API mutualisé via cache) — étend le payload
    _odds_raw_full = _fetch_odds_raw(fixture_id)
    _odds_parsed = _parse_all_odds(_odds_raw_full)
    fetched_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if market is not None:
        # Enrichit le market dict (rétro-compatible : champs sup ajoutés, anciens préservés)
        market.setdefault("fetched_at", fetched_now)
        market["bookmaker_id"]   = _odds_parsed.get("bookmaker_id")
        market["bookmaker_name"] = _odds_parsed.get("bookmaker")
        market["extended_markets"] = {
            "home":      _odds_parsed.get("home"),
            "draw":      _odds_parsed.get("draw"),
            "away":      _odds_parsed.get("away"),
            "over_25":   _odds_parsed.get("over_25"),
            "under_25":  _odds_parsed.get("under_25"),
            "over_15":   _odds_parsed.get("over_15"),
            "under_15":  _odds_parsed.get("under_15"),
            "btts_yes":  _odds_parsed.get("btts_yes"),
            "btts_no":   _odds_parsed.get("btts_no"),
            "double_1n": _odds_parsed.get("double_1n"),
            "double_12": _odds_parsed.get("double_12"),
            "double_n2": _odds_parsed.get("double_n2"),
        }
        market["all_bookmakers_raw"] = _odds_parsed.get("all_bookmakers", [])
    odds_fetched_at = fetched_now

    # ⑥ Jours de repos (depuis le dernier match)
    home_rest = _calc_rest_days(home_last, fixture_info.get("date"))
    away_rest = _calc_rest_days(away_last, fixture_info.get("date"))

    # ⑦ Lineups & Injuries
    lineups = None
    injuries = []
    home_missing = 0
    away_missing = 0
    home_missing_weighted = 0.0
    away_missing_weighted = 0.0
    home_formation_score = 0
    away_formation_score = 0
    try:
        lineups = fetch_lineups(fixture_id)
        injuries = fetch_injuries(fixture_id)
        home_missing = count_missing_players(injuries, home_id)
        away_missing = count_missing_players(injuries, away_id)

        # Récupérer les lineups par équipe (indexés par nom)
        home_lineup = None
        away_lineup = None
        if lineups:
            home_name = home_team.get("name")
            away_name = away_team.get("name")
            home_lineup = lineups.get(home_name)
            away_lineup = lineups.get(away_name)

        # Pondération des blessés selon statut dans les lineups
        home_missing_weighted = weighted_missing_score(injuries, home_id, home_lineup)
        away_missing_weighted = weighted_missing_score(injuries, away_id, away_lineup)

        # Score offensif de la formation
        if home_lineup:
            home_formation_score = formation_offensive_score(home_lineup.get("formation"))
        if away_lineup:
            away_formation_score = formation_offensive_score(away_lineup.get("formation"))
    except Exception as e:
        log.warning("Lineups/injuries indisponibles pour fixture %d : %s", fixture_id, e)

    # ⑧ Historique de compétition européenne (LDC / UEL / UECL)
    home_euro_history = []
    away_euro_history = []
    if league_id in EUROPEAN_CUP_IDS:
        log.info("  Coupe d'Europe detectee — fetch historique competition")
        home_euro_history = fetch_team_competition_history(home_id, league_id)
        away_euro_history = fetch_team_competition_history(away_id, league_id)

    # ⑨ Classement de la ligue (cache long via api_get)
    home_standing = None
    away_standing = None
    league_meta = None
    try:
        standings = fetch_team_standings(league_id, season)
        home_standing = standings.get(home_id)
        away_standing = standings.get(away_id)
        # Méta-données ligue pour features de motivation
        if standings:
            pts_list = sorted(
                [s.get("points", 0) for s in standings.values() if s.get("points") is not None],
                reverse=True
            )
            played_list = [s.get("played", 0) for s in standings.values() if s.get("played")]
            n_teams = len(pts_list)
            # Safety = 17e place en L1 (20 équipes), proportionnel pour ligues plus petites
            safety_idx = max(0, n_teams - 4) if n_teams > 4 else n_teams - 1
            league_meta = {
                "top_points": pts_list[0] if pts_list else 0,
                "safety_points": pts_list[safety_idx] if pts_list else 0,
                "max_played": max(played_list) if played_list else 0,
                "total_teams": n_teams,
            }
    except Exception as e:
        log.warning("Standings indisponibles pour league %d : %s", league_id, e)

    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "league_name": LEAGUES.get(league_id, {}).get("name", ""),
        "league_flag": LEAGUES.get(league_id, {}).get("flag", ""),
        "league_country": LEAGUES.get(league_id, {}).get("country", ""),
        "date": fixture_info.get("date"),
        "venue": fixture_info.get("venue", {}).get("name"),
        "is_european_cup": league_id in EUROPEAN_CUP_IDS,
        # ── Statut & score ──
        "match_status": match_status,
        "match_elapsed": match_elapsed,
        "current_home_goals": current_home_goals,
        "current_away_goals": current_away_goals,
        # ── Équipes ──
        "home_team": {
            "id": home_id,
            "name": home_team.get("name"),
            "logo": home_team.get("logo"),
        },
        "away_team": {
            "id": away_id,
            "name": away_team.get("name"),
            "logo": away_team.get("logo"),
        },
        "home_last_matches": home_last,
        "away_last_matches": away_last,
        "home_euro_history": home_euro_history,
        "away_euro_history": away_euro_history,
        "h2h": h2h,
        "referee": referee_data,
        "odds": market,
        "odd_over15": odd_over15,
        "odd_btts": odd_btts,
        # Étape 21.4 : timestamp anti-leakage (UTC ISO 8601)
        "odds_fetched_at": odds_fetched_at,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "lineups": lineups,
        "injuries": injuries,
        "home_missing_players": home_missing,
        "away_missing_players": away_missing,
        "home_missing_weighted": home_missing_weighted,
        "away_missing_weighted": away_missing_weighted,
        "home_formation_score": home_formation_score,
        "away_formation_score": away_formation_score,
        "home_standing": home_standing,
        "away_standing": away_standing,
        "league_meta": league_meta,
    }


# ══════════════════════════════════════════════
# 10. BATCH PIPELINE (toutes les ligues)
# ══════════════════════════════════════════════
def fetch_all_today() -> list[dict]:
    """
    Pipeline principal : récupère et enrichit tous les matchs du jour.
    Respecte le batch size et le quota.
    """
    log.info("═══ Smart Sim — Fetch du %s ═══", date.today().isoformat())
    log.info("Quota restant : %d / %d", quota.remaining, DAILY_QUOTA)

    # 1. Récupérer toutes les fixtures du jour
    all_fixtures = fetch_all_fixtures_today()
    total_matches = sum(len(v) for v in all_fixtures.values())
    log.info("Total matchs du jour : %d", total_matches)

    if total_matches == 0:
        log.info("Aucun match aujourd'hui dans les ligues configurées.")
        return []

    # 2. Enrichir chaque match (avec batch limit)
    enriched = []
    count = 0
    for league_id, fixtures in all_fixtures.items():
        season = LEAGUES[league_id]["season"]
        for fixture in fixtures:
            if count >= BATCH_SIZE:
                log.warning("Batch limit atteint (%d). Matchs restants ignorés.", BATCH_SIZE)
                break
            if not quota.can_call(5):  # Au moins 5 appels nécessaires par match
                log.warning("Quota trop bas pour continuer. Arrêt.")
                break

            match_data = fetch_full_match_data(fixture, league_id, season)
            if match_data:
                enriched.append(match_data)
                count += 1

        if count >= BATCH_SIZE or not quota.can_call(5):
            break

    log.info("═══ Fetch terminé : %d matchs enrichis ═══", len(enriched))
    log.info("Quota final : %d / %d utilisés", quota.used, DAILY_QUOTA)
    return enriched


def fetch_fixtures_by_date(league_id: int, season: int, target_date: date) -> list[dict]:
    """Récupère les fixtures pour une date spécifique (J+1, etc.)."""
    data = api_get("fixtures", {
        "league": league_id,
        "season": season,
        "date": target_date.isoformat(),
    }, ttl=CACHE_TTL_FIXTURES)
    if not data:
        return []
    return data.get("response", [])


def fetch_all_by_date(target_date: date, max_matches: int = None,
                      time_budget_seconds: int = 1200) -> list[dict]:
    """
    Pipeline complet pour une date arbitraire (ex: demain).
    Même logique que fetch_all_today mais avec date paramétrable.
    `max_matches`         : plafond optionnel (default = BATCH_SIZE de config).
    `time_budget_seconds` : budget wall-clock global (default 1200s = 20 min).
                            Si dépassé, retour partiel des matchs déjà enrichis.
    """
    import time as _time
    _t0 = _time.monotonic()
    _batch = max_matches if max_matches is not None else BATCH_SIZE
    log.info("═══ Smart Sim — Fetch du %s (batch=%d, budget=%ds) ═══",
             target_date.isoformat(), _batch, time_budget_seconds)
    log.info("Quota restant : %d / %d", quota.remaining, DAILY_QUOTA)

    all_fixtures = {}
    for league_id, meta in LEAGUES.items():
        if not quota.can_call():
            log.warning("Quota épuisé, arrêt du fetch fixtures.")
            break
        if _time.monotonic() - _t0 > time_budget_seconds:
            log.warning("Time budget dépassé pendant fetch fixtures.")
            break
        fixtures = fetch_fixtures_by_date(league_id, meta["season"], target_date)
        if fixtures:
            all_fixtures[league_id] = fixtures
            log.info("  %s %s : %d match(s)", meta["flag"], meta["name"], len(fixtures))

    total_matches = sum(len(v) for v in all_fixtures.values())
    log.info("Total matchs pour %s : %d", target_date.isoformat(), total_matches)

    if total_matches == 0:
        return []

    enriched = []
    count = 0
    timed_out = False
    for league_id, fixtures in all_fixtures.items():
        if timed_out:
            break
        season = LEAGUES[league_id]["season"]
        for fixture in fixtures:
            if count >= _batch:
                break
            if not quota.can_call(5):
                break
            if _time.monotonic() - _t0 > time_budget_seconds:
                log.warning("Time budget %ds dépassé après %d match(s) — retour partiel.",
                            time_budget_seconds, count)
                timed_out = True
                break
            match_data = fetch_full_match_data(fixture, league_id, season)
            if match_data:
                enriched.append(match_data)
                count += 1
        if count >= _batch or not quota.can_call(5):
            break

    elapsed = int(_time.monotonic() - _t0)
    log.info("═══ Fetch %s terminé : %d match(s) enrichi(s) en %ds%s ═══",
             target_date.isoformat(), len(enriched), elapsed,
             " (PARTIEL: timeout)" if timed_out else "")
    return enriched


# ══════════════════════════════════════════════
# CACHE DISQUE — Sauvegarde journalière JSON
# ══════════════════════════════════════════════
_DAILY_CACHE_DIR = Path(CACHE_DIR) / "daily"

def save_daily_cache(results: list[dict], target_date: date = None) -> Path:
    """
    Sauvegarde les résultats enrichis + prédictions dans un JSON daté.
    Fichier : cache/daily/cache_matchs_YYYY_MM_DD.json
    """
    _DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d = target_date or date.today()
    path = _DAILY_CACHE_DIR / f"cache_matchs_{d.isoformat()}.json"

    # Sérialiser — on doit gérer les types non-JSON
    def _serialize(obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, default=_serialize, indent=2)

    log.info("Cache journalier sauvegardé : %s (%d matchs)", path.name, len(results))
    return path


def load_daily_cache(target_date: date = None) -> Optional[list[dict]]:
    """
    Charge le cache JSON d'une journée donnée.
    Retourne None si le fichier n'existe pas.
    """
    d = target_date or date.today()
    path = _DAILY_CACHE_DIR / f"cache_matchs_{d.isoformat()}.json"

    if not path.exists():
        log.info("Pas de cache journalier pour le %s", d.isoformat())
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("Cache journalier chargé : %s (%d matchs)", path.name, len(data))
        return data
    except Exception as e:
        log.error("Erreur lecture cache %s : %s", path.name, e)
        return None


def list_cached_dates() -> list[date]:
    """Liste toutes les dates ayant un cache JSON, triées décroissant."""
    if not _DAILY_CACHE_DIR.exists():
        return []
    dates = []
    for f in _DAILY_CACHE_DIR.glob("cache_matchs_*.json"):
        try:
            d_str = f.stem.replace("cache_matchs_", "")
            dates.append(date.fromisoformat(d_str))
        except ValueError:
            continue
    return sorted(dates, reverse=True)


# ══════════════════════════════════════════════
# 11. CLASSEMENTS (cache long)
# ══════════════════════════════════════════════
def fetch_all_standings() -> dict[int, dict[int, dict]]:
    """Récupère les classements de toutes les ligues. {league_id: {team_id: standing}}."""
    all_standings = {}
    for league_id, meta in LEAGUES.items():
        standings = fetch_team_standings(league_id, meta["season"])
        if standings:
            all_standings[league_id] = standings
    return all_standings


# ══════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════
def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _calc_rest_days(last_matches: list[dict], next_match_date: Optional[str]) -> Optional[int]:
    """Calcule le nombre de jours depuis le dernier match joué."""
    if not last_matches or not next_match_date:
        return None

    try:
        next_dt = datetime.fromisoformat(next_match_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    # Le premier match de la liste est le plus récent
    for match in last_matches:
        match_date_str = match.get("fixture", {}).get("date")
        if match_date_str:
            try:
                match_dt = datetime.fromisoformat(match_date_str.replace("Z", "+00:00"))
                delta = (next_dt - match_dt).days
                if delta >= 0:
                    return delta
            except (ValueError, AttributeError):
                continue
    return None


def fetch_live_scores() -> dict[int, dict]:
    """
    Fetch ULTRA-LÉGER : récupère uniquement les fixtures du jour
    pour extraire score + statut + minute.
    UN appel par ligue (pas d'enrichissement, pas de stats, pas d'odds).
    Retourne {fixture_id: {status, elapsed, home_goals, away_goals}}.
    """
    today = date.today().isoformat()
    live_data = {}

    for league_id, meta in LEAGUES.items():
        if not quota.can_call():
            break

        # Forcer la fraîcheur : TTL très court (30s)
        key = _cache_key("fixtures", {"league": league_id, "season": meta["season"], "date": today})
        if key in _cache:
            del _cache[key]

        data = api_get("fixtures", {
            "league": league_id,
            "season": meta["season"],
            "date": today,
        }, ttl=30)  # Cache 30 secondes seulement

        if not data or not data.get("response"):
            continue

        for fix in data["response"]:
            fid = fix.get("fixture", {}).get("id")
            if fid:
                status_info = fix.get("fixture", {}).get("status", {})
                goals = fix.get("goals", {})
                live_data[fid] = {
                    "match_status": status_info.get("short", "NS"),
                    "match_elapsed": status_info.get("elapsed"),
                    "current_home_goals": goals.get("home"),
                    "current_away_goals": goals.get("away"),
                }

    log.info("Live scores : %d fixtures mises à jour.", len(live_data))
    return live_data


def clear_cache():
    """Vide le cache disque."""
    _cache.clear()
    log.info("Cache vidé.")


def clear_live_cache():
    """
    Vide uniquement le cache des fixtures du jour (pour refresh live).
    Ne touche pas au cache des stats historiques, odds, arbitres, etc.
    Parcourt toutes les clés et supprime celles liées aux fixtures du jour.
    """
    today = date.today().isoformat()
    keys_to_delete = []
    for key in _cache:
        try:
            # Les clés sont des hash SHA256, on ne peut pas filtrer par contenu.
            # Solution : on supprime toutes les entrées avec un TTL court (fixtures)
            pass
        except Exception:
            pass

    # Approche pragmatique : vider les fixtures en les re-fetchant sans cache
    # On expire manuellement les clés fixtures en les supprimant
    count = 0
    for league_id, meta in LEAGUES.items():
        params = {"league": league_id, "season": meta["season"], "date": today}
        key = _cache_key("fixtures", params)
        if key in _cache:
            del _cache[key]
            count += 1
    log.info("Cache live vidé : %d entrées fixtures supprimées.", count)


def get_quota_status() -> dict:
    """Retourne l'état actuel du quota."""
    return {
        "used": quota.used,
        "remaining": quota.remaining,
        "daily_limit": DAILY_QUOTA,
        "date": date.today().isoformat(),
    }
