"""Per-league scoring baselines — computed from historical_data.

For each league, we derive:
    lg_avg     : average goals scored per side per match
    home_bump  : multiplier applied to home lambda vs baseline
    away_bump  : multiplier applied to away lambda vs baseline

These replace the global constants (lg_avg=1.35, home_bump=1.10, away_bump=0.90)
in `_priors_from_form_v2` and `predict_from_ratings` for leagues with enough data.

Cache: in-memory, refreshed every REFRESH_SECONDS. Thread-safe read via GIL.

Fallback: if a league has <MIN_MATCHES samples in the window, returns global defaults.
"""
from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger("SmartSim.league_stats")

WINDOW_DAYS = 240      # 8 months rolling
MIN_MATCHES = 25       # below this, league keeps global defaults
REFRESH_SECONDS = 6 * 3600  # rebuild cache every 6h

DEFAULT_LG_AVG = 1.35
DEFAULT_HOME_BUMP = 1.10
DEFAULT_AWAY_BUMP = 0.90

_cache: dict = {"built_at": 0.0, "by_id": {}, "by_name": {}}
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        log.warning("supabase init failed: %s", e)
        return None


def _default() -> dict:
    return {
        "lg_avg": DEFAULT_LG_AVG,
        "home_bump": DEFAULT_HOME_BUMP,
        "away_bump": DEFAULT_AWAY_BUMP,
        "matches_seen": 0,
        "source": "default",
    }


def _build_cache() -> None:
    """Query historical_data, aggregate per league. Fully rebuilds the cache."""
    client = _get_client()
    if not client:
        _cache["built_at"] = time.time()
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).date().isoformat()
    try:
        # Pagination to avoid supabase 1000-row default cap
        rows = []
        page_size = 1000
        offset = 0
        while True:
            resp = (client.table("historical_data")
                        .select("league_id, league_name, home_goals, away_goals, date")
                        .gte("date", cutoff)
                        .range(offset, offset + page_size - 1)
                        .execute())
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
            if offset > 50000:  # safety
                break
    except Exception as e:
        log.warning("historical_data query failed: %s", e)
        _cache["built_at"] = time.time()
        return

    # Aggregate
    per_id: dict = {}
    per_name: dict = {}
    for r in rows:
        hg, ag = r.get("home_goals"), r.get("away_goals")
        if hg is None or ag is None:
            continue
        try:
            hg = int(hg); ag = int(ag)
        except (TypeError, ValueError):
            continue
        lid = r.get("league_id")
        lname = (r.get("league_name") or "").strip()
        for key, bucket in ((lid, per_id), (lname, per_name)):
            if not key:
                continue
            entry = bucket.setdefault(key, {"h": 0.0, "a": 0.0, "n": 0})
            entry["h"] += hg
            entry["a"] += ag
            entry["n"] += 1

    def _finalize(bucket: dict) -> dict:
        out = {}
        for key, e in bucket.items():
            n = e["n"]
            if n < MIN_MATCHES:
                continue
            avg_h = e["h"] / n
            avg_a = e["a"] / n
            lg_avg = max(0.8, min(2.2, (avg_h + avg_a) / 2))
            home_bump = max(0.85, min(1.35, avg_h / lg_avg if lg_avg > 0 else 1.10))
            away_bump = max(0.75, min(1.15, avg_a / lg_avg if lg_avg > 0 else 0.90))
            out[key] = {
                "lg_avg": lg_avg,
                "home_bump": home_bump,
                "away_bump": away_bump,
                "matches_seen": n,
                "source": "league",
            }
        return out

    _cache["by_id"] = _finalize(per_id)
    _cache["by_name"] = _finalize(per_name)
    _cache["built_at"] = time.time()
    log.info("league_stats cache built: %d leagues by id, %d by name (window %dd)",
             len(_cache["by_id"]), len(_cache["by_name"]), WINDOW_DAYS)


def _maybe_refresh() -> None:
    if time.time() - _cache["built_at"] > REFRESH_SECONDS:
        _build_cache()


def get_baseline(league_id=None, league_name: str = "") -> dict:
    """Return per-league baseline, or defaults if not enough data."""
    _maybe_refresh()
    if league_id:
        try:
            found = _cache["by_id"].get(int(league_id))
            if found:
                return found
        except (TypeError, ValueError):
            pass
    if league_name:
        found = _cache["by_name"].get(league_name.strip())
        if found:
            return found
    return _default()


def stats_summary() -> dict:
    """Diagnostic: how many leagues are known with per-league baselines."""
    _maybe_refresh()
    return {
        "built_at": _cache["built_at"],
        "leagues_by_id": len(_cache["by_id"]),
        "leagues_by_name": len(_cache["by_name"]),
        "window_days": WINDOW_DAYS,
        "min_matches": MIN_MATCHES,
        "sample_top": [
            {"key": str(k), **v}
            for k, v in sorted(
                _cache["by_id"].items(),
                key=lambda kv: kv[1]["matches_seen"],
                reverse=True,
            )[:10]
        ],
    }
