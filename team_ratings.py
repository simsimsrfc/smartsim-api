"""Persistent Elo-lite team ratings — Supabase-backed.

Rating format per team:
    attack  ∈ [0.3, 2.5]   # goals scoring propensity vs baseline
    defense ∈ [0.3, 2.5]   # goals conceding propensity vs baseline

Updated after every finished match via `update_ratings_from_match`.
Predictions read via `get_ratings(team_ids)`.

Walk-forward safe: only past FT matches are folded in (via match_ledger).
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("SmartSim.ratings")

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
ALPHA = 0.12
ATT_MIN, ATT_MAX = 0.3, 2.5
DEF_MIN, DEF_MAX = 0.3, 2.5
HOME_BUMP = 1.10
AWAY_BUMP = 0.90
DEFAULT_LG_AVG = 1.35

_client = None

def _get_client():
    global _client
    if _client is not None: return _client
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not (url and key): return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        log.warning("Supabase client init échoué : %s", e)
        return None


def get_ratings(team_ids: list[int]) -> dict[int, dict]:
    """Retourne {team_id: {attack, defense, matches_seen}} pour les IDs demandés."""
    if not team_ids: return {}
    client = _get_client()
    if not client: return {}
    try:
        resp = (client.table("team_ratings")
                    .select("team_id, attack, defense, matches_seen")
                    .in_("team_id", list(set(team_ids)))
                    .execute())
        out = {}
        for row in (resp.data or []):
            out[int(row["team_id"])] = {
                "attack": float(row.get("attack") or 1.0),
                "defense": float(row.get("defense") or 1.0),
                "matches_seen": int(row.get("matches_seen") or 0),
            }
        return out
    except Exception as e:
        log.warning("get_ratings failed: %s", e)
        return {}


def _upsert_rating(team_id: int, team_name: str, attack: float, defense: float, matches_seen: int) -> bool:
    client = _get_client()
    if not client: return False
    try:
        client.table("team_ratings").upsert({
            "team_id": int(team_id),
            "team_name": team_name or "",
            "attack": max(ATT_MIN, min(ATT_MAX, float(attack))),
            "defense": max(DEF_MIN, min(DEF_MAX, float(defense))),
            "matches_seen": int(matches_seen),
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, on_conflict="team_id").execute()
        return True
    except Exception as e:
        log.warning("upsert_rating %s failed: %s", team_id, e)
        return False


def _is_processed(fixture_id: str) -> bool:
    client = _get_client()
    if not client: return False
    try:
        r = (client.table("match_ledger").select("fixture_id")
                 .eq("fixture_id", str(fixture_id)).limit(1).execute())
        return bool(r.data)
    except Exception:
        return False


def _mark_processed(fixture_id: str, match_date: str, hid: int, aid: int, hg: int, ag: int):
    client = _get_client()
    if not client: return
    try:
        client.table("match_ledger").upsert({
            "fixture_id": str(fixture_id),
            "match_date": match_date,
            "home_id": int(hid), "away_id": int(aid),
            "home_goals": int(hg), "away_goals": int(ag),
        }, on_conflict="fixture_id").execute()
    except Exception as e:
        log.warning("mark_processed %s failed: %s", fixture_id, e)


def update_ratings_from_match(fixture_id: str, match_date: str,
                                  home_id: int, home_name: str, home_goals: int,
                                  away_id: int, away_name: str, away_goals: int) -> bool:
    """Update Elo ratings from a FT match. Idempotent via match_ledger."""
    if _is_processed(fixture_id):
        return False
    # Load current ratings
    current = get_ratings([home_id, away_id])
    hr = current.get(home_id, {"attack": 1.0, "defense": 1.0, "matches_seen": 0})
    ar = current.get(away_id, {"attack": 1.0, "defense": 1.0, "matches_seen": 0})
    # Expected goals from ratings
    exp_h = hr["attack"] * ar["defense"] * HOME_BUMP
    exp_a = ar["attack"] * hr["defense"] * AWAY_BUMP
    # Update
    hr["attack"]  = max(ATT_MIN, min(ATT_MAX, hr["attack"]  + ALPHA * (home_goals - exp_h)))
    ar["defense"] = max(DEF_MIN, min(DEF_MAX, ar["defense"] + ALPHA * (home_goals - exp_h) * 0.5))
    ar["attack"]  = max(ATT_MIN, min(ATT_MAX, ar["attack"]  + ALPHA * (away_goals - exp_a)))
    hr["defense"] = max(DEF_MIN, min(DEF_MAX, hr["defense"] + ALPHA * (away_goals - exp_a) * 0.5))
    _upsert_rating(home_id, home_name, hr["attack"], hr["defense"], hr["matches_seen"] + 1)
    _upsert_rating(away_id, away_name, ar["attack"], ar["defense"], ar["matches_seen"] + 1)
    _mark_processed(fixture_id, match_date, home_id, away_id, home_goals, away_goals)
    return True


def predict_from_ratings(home_id: int, away_id: int, lg_avg: float = DEFAULT_LG_AVG) -> Optional[dict]:
    """Retourne {home, draw, away, over_25, over_15, btts} depuis les ratings.
    Retourne None si aucune donnée pour les deux équipes."""
    import math
    ratings = get_ratings([home_id, away_id])
    if not ratings: return None
    hr = ratings.get(home_id, {"attack": 1.0, "defense": 1.0, "matches_seen": 0})
    ar = ratings.get(away_id, {"attack": 1.0, "defense": 1.0, "matches_seen": 0})
    # Si aucune équipe n'a été vue → priors trop faibles pour être utiles
    if hr["matches_seen"] == 0 and ar["matches_seen"] == 0:
        return None
    lam_h = max(0.3, min(3.8, hr["attack"] * ar["defense"] * lg_avg * HOME_BUMP))
    lam_a = max(0.2, min(3.5, ar["attack"] * hr["defense"] * lg_avg * AWAY_BUMP))
    def pg(l, k): return math.exp(-l) * l**k / math.factorial(k)
    ph = pd = pa = po25 = po15 = pbtts = 0.0
    for i in range(7):
        for j in range(7):
            p = pg(lam_h, i) * pg(lam_a, j)
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
            if i+j >= 3: po25 += p
            if i+j >= 2: po15 += p
            if i >= 1 and j >= 1: pbtts += p
    return {"home": ph, "draw": pd, "away": pa,
              "over_25": po25, "over_15": po15, "btts": pbtts,
              "matches_seen": min(hr["matches_seen"], ar["matches_seen"])}
