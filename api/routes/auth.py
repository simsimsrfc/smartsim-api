"""
Routes /api/auth/*

POST /api/auth/signup       — Créer un compte (email + password)
POST /api/auth/login        — Se connecter, retourne JWT
POST /api/auth/logout       — Invalide la session côté Supabase
GET  /api/auth/me           — Infos utilisateur courant (auth requise)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.deps import get_supabase, get_current_user

log = logging.getLogger("SmartSim.api.auth")
router = APIRouter()


# ──────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────
class SignupBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    user: dict


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _extract_session(resp) -> dict:
    """Extrait session + user d'une réponse Supabase (objet ou dict)."""
    session = getattr(resp, "session", None) or (resp.get("session") if isinstance(resp, dict) else None)
    user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)
    if not session or not user:
        raise HTTPException(401, "Réponse Supabase invalide (session ou user manquant).")

    access_token = getattr(session, "access_token", None) or (session.get("access_token") if isinstance(session, dict) else None)
    refresh_token = getattr(session, "refresh_token", None) or (session.get("refresh_token") if isinstance(session, dict) else None)
    expires_in = getattr(session, "expires_in", None) or (session.get("expires_in") if isinstance(session, dict) else None)

    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    user_email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)

    return {
        "access_token": access_token or "",
        "refresh_token": refresh_token,
        "expires_in": expires_in,
        "user": {"id": str(user_id), "email": user_email or ""},
    }


# ══════════════════════════════════════════════════════════════
# POST /api/auth/signup
# ══════════════════════════════════════════════════════════════
@router.post("/signup", summary="Créer un compte")
async def signup(body: SignupBody):
    client = get_supabase()
    try:
        resp = client.auth.sign_up({
            "email": body.email,
            "password": body.password,
        })
    except Exception as e:
        msg = str(e).lower()
        if "already" in msg or "registered" in msg or "exists" in msg:
            raise HTTPException(409, "Un compte existe déjà avec cet email.")
        log.error("Erreur signup : %s", e)
        raise HTTPException(400, f"Inscription impossible : {e}")

    # Si confirm_email est activé, session peut être null ; on retourne un état
    session = getattr(resp, "session", None) or (resp.get("session") if isinstance(resp, dict) else None)
    user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)

    if session is None and user is not None:
        # Compte créé mais pas de session (email à confirmer ou autoconfirm off)
        u_id = getattr(user, "id", None) or user.get("id") if isinstance(user, dict) else None
        u_email = getattr(user, "email", None) or user.get("email") if isinstance(user, dict) else None
        return {
            "status": "pending_confirmation",
            "message": "Compte créé. Vérifie tes emails pour confirmer.",
            "user": {"id": str(u_id) if u_id else None, "email": u_email},
        }

    return {"status": "ok", **_extract_session(resp)}


# ══════════════════════════════════════════════════════════════
# POST /api/auth/login
# ══════════════════════════════════════════════════════════════
@router.post("/login", summary="Se connecter (email + password)")
async def login(body: LoginBody):
    client = get_supabase()
    try:
        resp = client.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
    except Exception as e:
        msg = str(e).lower()
        if "invalid" in msg or "credentials" in msg or "password" in msg:
            raise HTTPException(401, "Email ou mot de passe incorrect.")
        log.error("Erreur login : %s", e)
        raise HTTPException(401, "Connexion impossible.")

    return {"status": "ok", **_extract_session(resp)}


# ══════════════════════════════════════════════════════════════
# POST /api/auth/logout
# ══════════════════════════════════════════════════════════════
@router.post("/logout", summary="Se déconnecter (invalide la session)")
async def logout(user: dict = Depends(get_current_user)):
    client = get_supabase()
    try:
        # Supabase a une méthode admin pour invalider la session
        client.auth.sign_out()
    except Exception as e:
        log.warning("sign_out a échoué (non bloquant) : %s", e)
    return {"status": "ok", "message": "Déconnecté."}


# ══════════════════════════════════════════════════════════════
# GET /api/auth/me
# ══════════════════════════════════════════════════════════════
@router.get("/me", summary="Infos de l'utilisateur courant")
async def me(user: dict = Depends(get_current_user)):
    return {"user": user}
