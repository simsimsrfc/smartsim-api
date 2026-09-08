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

    # Refresh du cache journalier en tâche de fond au démarrage
    if os.environ.get("SMARTSIM_STARTUP_REFRESH", "1") == "1":
        import threading
        def _bg_refresh():
            try:
                from data_fetcher import load_daily_cache, save_daily_cache, fetch_all_by_date
                from datetime import date
                try:
                    from zoneinfo import ZoneInfo
                    today = __import__("datetime").datetime.now(ZoneInfo("Europe/Paris")).date()
                except Exception:
                    today = date.today()
                if load_daily_cache(today):
                    log.info("[startup-refresh] cache déjà présent pour %s, skip", today)
                    return
                log.info("[startup-refresh] lancement fetch pour %s", today)
                from simbet_v2_bridge import predict_today_v2 as _predict
                from extra_markets import compute_extra_markets
                raw = fetch_all_by_date(today)
                if raw:
                    results = _predict(raw)
                    for m in results:
                        try: m["extra_markets"] = compute_extra_markets(m)
                        except Exception: pass
                    save_daily_cache(results, target_date=today)
                    log.info("[startup-refresh] cache sauvegardé : %d matchs", len(results))
            except Exception as e:
                log.warning("[startup-refresh] échec : %s", e)
        threading.Thread(target=_bg_refresh, daemon=True).start()

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
