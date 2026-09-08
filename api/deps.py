"""
Dépendances FastAPI partagées :
- get_supabase()              → client Supabase admin (service_role)
- get_current_user()          → vérifie le JWT Supabase, retourne le user
- get_current_user_optional() → variante non-bloquante

Le schéma `HTTPBearer` rend visible le bouton 🔓 Authorize dans Swagger
et fait apparaître un cadenas sur chaque route protégée.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("SmartSim.api.deps")


# ──────────────────────────────────────────────
# Schéma de sécurité (visible dans /api/docs)
# ──────────────────────────────────────────────
# auto_error=False  → on gère nous-mêmes le 401 dans get_current_user,
#                     ce qui permet d'avoir une variante optionnelle propre.
bearer_scheme = HTTPBearer(
    bearerFormat="JWT",
    description="Colle ici le `access_token` retourné par /api/auth/login.",
    auto_error=False,
)


# ──────────────────────────────────────────────
# Client Supabase admin (singleton)
# ──────────────────────────────────────────────
_admin_client = None


def get_supabase():
    """
    Retourne un client Supabase configuré avec service_role.
    Utilisé pour les opérations CRUD côté serveur (favorites, bet_history…)
    qui by-passent la RLS.
    """
    global _admin_client
    if _admin_client is not None:
        return _admin_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(503, "Supabase non configuré (env SUPABASE_URL / SUPABASE_KEY).")

    try:
        from supabase import create_client
        _admin_client = create_client(url, key)
        return _admin_client
    except Exception as e:
        log.error("Erreur init Supabase : %s", e)
        raise HTTPException(503, f"Supabase indisponible : {e}")


# ──────────────────────────────────────────────
# Auth dependency : vérification JWT Supabase
# ──────────────────────────────────────────────
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    Vérifie le token JWT Supabase passé en `Authorization: Bearer <jwt>`.
    Retourne {id, email} si valide, sinon raise 401.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(401, "Token manquant. Clique 🔓 Authorize dans Swagger et colle ton access_token.")

    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(401, "Token vide.")

    client = get_supabase()
    try:
        user_resp = client.auth.get_user(token)
        user = getattr(user_resp, "user", None) or (user_resp.get("user") if isinstance(user_resp, dict) else None)
    except Exception as e:
        log.warning("get_user() Supabase a échoué : %s", e)
        raise HTTPException(401, "Token invalide ou expiré.")

    if user is None:
        raise HTTPException(401, "Token non reconnu.")

    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    user_email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)

    if not user_id:
        raise HTTPException(401, "User ID manquant dans le token.")

    return {"id": str(user_id), "email": user_email or ""}


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[dict]:
    """Variante qui retourne None si pas auth (pas de 401)."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None
