"""
Smart Sim — Backend API (FastAPI)
Expose les fonctions métier existantes (model.py, features.py, data_fetcher.py,
supabase_db.py) via une API REST consommée par le futur frontend Next.js.

Run en local :
    uvicorn api.main:app --reload --port 8000

Run en prod (Hugging Face Space Docker) :
    uvicorn api.main:app --host 0.0.0.0 --port 7860
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ──────────────────────────────────────────────
# 1. CHEMINS — racine du projet dans sys.path
# ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ──────────────────────────────────────────────
# 2. CHARGEMENT .env AVANT TOUT IMPORT MÉTIER
# ──────────────────────────────────────────────
# Il faut que SUPABASE_URL / SUPABASE_KEY / FOOTBALL_API_KEY soient
# disponibles avant qu'on importe deps.py (qui appelle os.environ.get).
# En prod (Hugging Face Space), les variables viennent des "Secrets" du
# Space — pas de .env nécessaire. En local, le .env à la racine est lu.
try:
    from dotenv import load_dotenv
    _ENV_PATH = _ROOT / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    # python-dotenv absent : on continue, les variables doivent venir
    # de l'environnement système (HF Secrets en prod).
    pass

from fastapi import FastAPI                      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.routes import matches, history, auth, favorites  # noqa: E402

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("SmartSim.api")


# ──────────────────────────────────────────────
# LIFESPAN (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("=== Smart Sim API démarrée ===")
    log.info("Routes montées : /api/auth/* · /api/matches/* · /api/history/* · /api/favorites/*")

    # Vérification des variables d'environnement critiques
    # (on n'affiche JAMAIS les valeurs, juste leur présence)
    env_status = {
        "SUPABASE_URL": "✅ ok" if os.environ.get("SUPABASE_URL") else "❌ manquant",
        "SUPABASE_KEY": "✅ ok" if os.environ.get("SUPABASE_KEY") else "❌ manquant",
        "FOOTBALL_API_KEY": "✅ ok" if os.environ.get("FOOTBALL_API_KEY") else "❌ manquant",
    }
    for name, status in env_status.items():
        log.info("ENV %s : %s", name, status)

    # Refresh du cache journalier en tâche de fond
    # - au démarrage : fetch J+0 et J+1
    # - chaque nuit à 00:15 Paris : fetch J+0 (nouveau jour) et J+1
    if os.environ.get("SMARTSIM_STARTUP_REFRESH", "1") == "1":
        import threading, time as _time
        from datetime import date, datetime, timedelta

        _refresh_lock = threading.Lock()

        def _paris_now():
            try:
                from zoneinfo import ZoneInfo
                return datetime.now(ZoneInfo("Europe/Paris"))
            except Exception:
                return datetime.now()

        def _refresh_one(target):
            from data_fetcher import load_daily_cache, save_daily_cache, fetch_all_by_date
            from simbet_v2_bridge import predict_today_v2 as _predict
            from extra_markets import compute_extra_markets
            if load_daily_cache(target):
                log.info("[refresh] cache déjà présent pour %s, skip", target)
                return
            log.info("[refresh] lancement fetch pour %s", target)
            raw = fetch_all_by_date(target)
            if not raw:
                log.info("[refresh] aucun match pour %s", target)
                return
            results = _predict(raw)
            for m in results:
                try: m["extra_markets"] = compute_extra_markets(m)
                except Exception: pass
            save_daily_cache(results, target_date=target)
            log.info("[refresh] cache sauvegardé pour %s : %d matchs", target, len(results))
            # Persistance historique Supabase (pour l'onglet Historique)
            try:
                from supabase_db import save_bet_history
                n = save_bet_history(results, target_date=target.isoformat())
                log.info("[refresh] bet_history Supabase : %s ligne(s) pour %s", n, target)
            except Exception as e:
                log.warning("[refresh] save_bet_history %s : %s", target, e)

        def _refresh_two_days():
            if not _refresh_lock.acquire(blocking=False):
                log.info("[refresh] déjà en cours, skip")
                return
            try:
                today = _paris_now().date()
                for d in (today, today + timedelta(days=1)):
                    try: _refresh_one(d)
                    except Exception as e: log.warning("[refresh] %s échec : %s", d, e)
            finally:
                _refresh_lock.release()

        def _update_elo_from_recent_ft(days_back: int = 7):
            """Scanne les caches journaliers récents et incorpore chaque match FT
            pas encore présent dans match_ledger dans les ratings Elo persistants.
            Idempotent grâce à match_ledger."""
            try:
                from data_fetcher import load_daily_cache
                from team_ratings import update_ratings_from_match
            except Exception as e:
                log.warning("[elo-update] imports échoués : %s", e); return
            today = _paris_now().date()
            updated = seen = 0
            for i in range(days_back):
                d = today - timedelta(days=i)
                cached = load_daily_cache(d)
                if not cached: continue
                for m in cached:
                    status = ((m.get("match_status") or "")).upper()
                    if status not in ("FT", "AET", "PEN"): continue
                    hg, ag = m.get("current_home_goals"), m.get("current_away_goals")
                    if hg is None or ag is None: continue
                    home, away = m.get("home_team") or {}, m.get("away_team") or {}
                    hid, aid = home.get("id"), away.get("id")
                    fix_id = str(m.get("fixture_id") or "")
                    if not (hid and aid and fix_id): continue
                    try:
                        if update_ratings_from_match(
                            fix_id, m.get("date") or d.isoformat(),
                            int(hid), home.get("name") or "", int(hg),
                            int(aid), away.get("name") or "", int(ag)):
                            updated += 1
                        else:
                            seen += 1
                    except Exception as e:
                        log.warning("[elo-update] fold %s : %s", fix_id, e)
            log.info("[elo-update] %d nouveaux matchs foldés (%d déjà connus, %d jours scannés)",
                     updated, seen, days_back)

        def _midnight_scheduler():
            while True:
                now = _paris_now()
                # Prochain déclenchement : 00:15 Paris le lendemain
                nxt = (now + timedelta(days=1)).replace(hour=0, minute=15, second=0, microsecond=0)
                sleep_s = max(60, (nxt - now).total_seconds())
                log.info("[scheduler] prochain refresh à %s (dans %.0fs)", nxt, sleep_s)
                _time.sleep(sleep_s)
                try: _refresh_two_days()
                except Exception as e: log.warning("[scheduler] tick échec : %s", e)
                # Après le refresh J+0/J+1, fold les FT récents dans Elo
                try: _update_elo_from_recent_ft(days_back=3)
                except Exception as e: log.warning("[elo-update] tick : %s", e)

        def _elo_hourly_scheduler():
            """Toutes les 3h, fold les FT récents dans Elo (matches finissent à des horaires variés)."""
            while True:
                _time.sleep(3 * 3600)  # 3h
                try: _update_elo_from_recent_ft(days_back=2)
                except Exception as e: log.warning("[elo-hourly] tick : %s", e)

        def _results_sync_scheduler():
            """Toutes les 2h, sync les résultats FT depuis API-Football vers bet_history."""
            while True:
                _time.sleep(2 * 3600)  # 2h
                try:
                    from api.results_sync import sync_pending_results
                    report = sync_pending_results(max_api_calls=40, days_back=5)
                    log.info("[results-sync] %s", report)
                except Exception as e:
                    log.warning("[results-sync] tick : %s", e)

        threading.Thread(target=_refresh_two_days, daemon=True).start()
        threading.Thread(target=_midnight_scheduler, daemon=True).start()
        # Fold Elo au startup + toutes les 3h
        threading.Thread(target=lambda: _update_elo_from_recent_ft(days_back=7), daemon=True).start()
        threading.Thread(target=_elo_hourly_scheduler, daemon=True).start()
        # Sync résultats FT toutes les 2h (peuple result_*_won dans bet_history)
        threading.Thread(target=_results_sync_scheduler, daemon=True).start()

    yield
    log.info("=== Smart Sim API arrêtée ===")


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = FastAPI(
    title="Smart Sim API",
    description="API officielle Smart Sim — analyses, prédictions et historiques de matchs.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# ──────────────────────────────────────────────
# CORS — autorise le frontend Next.js (Vercel) à appeler l'API
# ──────────────────────────────────────────────
# En dev : localhost:3000 (Next.js dev server)
# En prod : ton domaine Vercel
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
_extra_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_extra_origins = [o.strip() for o in _extra_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(matches.router, prefix="/api/matches", tags=["matches"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(favorites.router, prefix="/api/favorites", tags=["favorites"])


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
@app.get("/api/health", tags=["meta"])
async def health():
    """Endpoint de santé pour monitoring + debug rapide."""
    return {
        "status": "ok",
        "service": "smart-sim-api",
        "version": "0.1.0",
    }


@app.get("/", tags=["meta"])
async def root():
    """Racine — redirige vers la doc Swagger."""
    return {
        "service": "Smart Sim API",
        "docs": "/api/docs",
        "health": "/api/health",
    }
