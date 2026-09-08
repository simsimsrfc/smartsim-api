# ══════════════════════════════════════════════════════════════
# Smart Sim API — Dockerfile pour Hugging Face Space (mode Docker)
# Expose FastAPI sur le port 7860 (port standard HF Spaces)
# ══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Variables d'env Python (-u : flush logs immédiatement, pas de .pyc)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dépendances système minimales (xgboost / lightgbm ont besoin de libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────
# Working directory + permissions
# (HF Spaces tournent en utilisateur non-root par défaut)
# ──────────────────────────────────────────────
WORKDIR /app

# ──────────────────────────────────────────────
# 1) Installation des dépendances (en couche séparée pour cache Docker)
# ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ──────────────────────────────────────────────
# 2) Copie du code source
#    On copie tout (api/, model.py, features.py, data_fetcher.py,
#    supabase_db.py, auth.py, config.py). Streamlit (app.py) est inclus
#    mais NON démarré ici — ce conteneur sert uniquement à FastAPI.
# ──────────────────────────────────────────────
COPY . .

# Créer les dossiers de cache utilisés par data_fetcher / model
RUN mkdir -p /app/cache /app/cache/daily /app/models

# ──────────────────────────────────────────────
# Exposer le port 7860 (standard HF Spaces)
# ──────────────────────────────────────────────
EXPOSE 7860

# ──────────────────────────────────────────────
# Lancement de l'API
# - host 0.0.0.0  : accessible depuis l'extérieur du conteneur
# - port 7860     : port HF Spaces par défaut
# - workers 1     : 1 worker suffit pour démarrer (HF free tier 2 vCPU)
# - log-level info: équilibre verbosité / clarté
# ──────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
