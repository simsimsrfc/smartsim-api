"""Cache team_name → logo_url — utilisé par load_bet_history pour hydrater les logos
qui ne sont pas stockés dans la table bet_history.

Source : historical_data.full_data (déjà persisté, contient home_team/away_team avec logos).
Cache RAM, refresh 6h.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional

log = logging.getLogger("SmartSim.team_logos")

REFRESH_SECONDS = 6 * 3600
MAX_ROWS_TO_SCAN = 3000

_cache: dict = {"built_at": 0.0, "by_name": {}, "by_id": {}}
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


def _build_cache() -> None:
    client = _get_client()
    if not client:
        _cache["built_at"] = time.time()
        return
    by_name: dict[str, str] = {}
    by_id: dict[int, str] = {}
    offset = 0
    step = 500
    scanned = 0
    try:
        while scanned < MAX_ROWS_TO_SCAN:
            resp = (client.table("historical_data")
                        .select("full_data")
                        .order("date", desc=True)
                        .range(offset, offset + step - 1)
                        .execute())
            rows = resp.data or []
            if not rows:
                break
            for row in rows:
                fd = row.get("full_data") or {}
                for side in ("home_team", "away_team"):
                    t = fd.get(side) or {}
                    name = (t.get("name") or "").strip()
                    logo = (t.get("logo") or "").strip()
                    tid = t.get("id")
                    if name and logo and name not in by_name:
                        by_name[name] = logo
                    if tid and logo:
                        try:
                            by_id[int(tid)] = logo
                        except (TypeError, ValueError):
                            pass
            scanned += len(rows)
            if len(rows) < step:
                break
            offset += step
    except Exception as e:
        log.warning("team_logos build failed: %s", e)
    _cache["by_name"] = by_name
    _cache["by_id"] = by_id
    _cache["built_at"] = time.time()
    log.info("team_logos cache built: %d names, %d ids", len(by_name), len(by_id))


def _maybe_refresh() -> None:
    if time.time() - _cache["built_at"] > REFRESH_SECONDS:
        _build_cache()


def get_logo(team_name: str = "", team_id=None) -> str:
    """Retourne l'URL du logo (ou chaîne vide si inconnu)."""
    _maybe_refresh()
    if team_id is not None:
        try:
            found = _cache["by_id"].get(int(team_id))
            if found:
                return found
        except (TypeError, ValueError):
            pass
    if team_name:
        found = _cache["by_name"].get(team_name.strip())
        if found:
            return found
    return ""
