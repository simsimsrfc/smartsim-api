"""
Smart Sim — Authentification & Gestion des utilisateurs
Stockage Supabase (table users) + fallback local pour mode hors-ligne.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("SmartSim.auth")

# ══════════════════════════════════════════════
# ADMINISTRATEUR UNIQUE
# ══════════════════════════════════════════════
ADMIN_EMAIL = "Simsimsrfc@gmail.com"
ADMIN_PASSWORD = "simbetsport2026"

# ══════════════════════════════════════════════
# STOCKAGE LOCAL (fallback si Supabase indisponible)
# ══════════════════════════════════════════════
_USERS_LOCAL_FILE = Path(__file__).parent / "users_local.json"


# ══════════════════════════════════════════════
# HASHAGE SECURISE (SHA-256 + salt aleatoire)
# ══════════════════════════════════════════════
def _hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Hash un mot de passe avec SHA-256 + salt.
    Format retourne : {salt_32_hex}${hash_64_hex}
    """
    if salt is None:
        salt = os.urandom(16).hex()
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${h}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verifie un mot de passe contre son hash stocke."""
    if not stored_hash or not isinstance(stored_hash, str):
        return False
    if "$" not in stored_hash:
        return False
    try:
        salt, _ = stored_hash.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return _hash_password(password, salt) == stored_hash


# ══════════════════════════════════════════════
# SUPABASE — Connexion via le client partagé
# ══════════════════════════════════════════════
def _get_supabase():
    """Récupère le client Supabase singleton."""
    try:
        from supabase_db import _get_client
        return _get_client()
    except Exception as e:
        log.warning("Supabase indisponible : %s", e)
        return None


# ══════════════════════════════════════════════
# STOCKAGE LOCAL (fallback)
# ══════════════════════════════════════════════
def _load_users_local() -> dict:
    if _USERS_LOCAL_FILE.exists():
        try:
            with open(_USERS_LOCAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Fichier users_local corrompu : %s", e)
    return {}


def _save_users_local(users: dict) -> None:
    try:
        with open(_USERS_LOCAL_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error("Erreur sauvegarde users_local : %s", e)


# ══════════════════════════════════════════════
# CHARGEMENT DES UTILISATEURS
# ══════════════════════════════════════════════
def load_users() -> dict:
    """
    Charge tous les utilisateurs depuis Supabase.
    Fallback sur le fichier local si Supabase indisponible.
    Retourne {email_lowercase: user_dict}.
    """
    client = _get_supabase()
    if client is None:
        log.warning("Supabase indisponible — chargement local")
        return _load_users_local()

    try:
        resp = client.table("users").select("*").execute()
        rows = resp.data or []
        users = {}
        for row in rows:
            email = str(row.get("email", "")).strip()
            if not email:
                continue
            users[email.lower()] = {
                "email": email,
                "password_hash": str(row.get("password_hash", "")),
                "created_at": str(row.get("created_at", "")),
            }
        log.info("Users charges via Supabase : %d comptes", len(users))
        return users
    except Exception as e:
        log.warning("Erreur lecture users Supabase : %s — fallback local", e)
        return _load_users_local()


# ══════════════════════════════════════════════
# CREATION D'UN COMPTE
# ══════════════════════════════════════════════
def save_user(email: str, password: str) -> tuple[bool, str]:
    """
    Cree un nouveau compte utilisateur.
    Retourne (success, message).
    """
    email_clean = email.strip()
    email_key = email_clean.lower()

    if not email_clean or "@" not in email_clean:
        return False, "Email invalide"
    if len(password) < 6:
        return False, "Mot de passe trop court (6 caracteres minimum)"
    if email_key == ADMIN_EMAIL.lower():
        return False, "Cet email est reserve"

    users = load_users()
    if email_key in users:
        return False, "Un compte existe deja avec cet email"

    pw_hash = _hash_password(password)
    new_user = {
        "email": email_clean,
        "password_hash": pw_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    log.info("Creation compte : %s", email_clean)

    # Tentative d'ecriture Supabase
    client = _get_supabase()
    if client is not None:
        try:
            client.table("users").insert(new_user).execute()
            log.info("Utilisateur cree sur Supabase : %s", email_clean)
            # Ecrire aussi en local pour fallback offline
            users[email_key] = new_user
            _save_users_local(users)
            return True, "Compte cree avec succes"
        except Exception as e:
            log.warning("Erreur ecriture Supabase : %s — fallback local", e)

    # Fallback : stockage local uniquement
    users[email_key] = new_user
    _save_users_local(users)
    log.info("Utilisateur cree en local (Supabase indisponible) : %s", email_clean)
    return True, "Compte cree avec succes"


# ══════════════════════════════════════════════
# AUTHENTIFICATION
# ══════════════════════════════════════════════
def authenticate(email: str, password: str) -> Optional[dict]:
    """
    Verifie email + password.
    Retourne le user dict si OK, None sinon.
    """
    email_clean = email.strip()
    email_key = email_clean.lower()

    if not email_clean or not password:
        return None

    # Cas admin (hardcoded, pas de hash)
    if email_key == ADMIN_EMAIL.lower():
        if password == ADMIN_PASSWORD:
            log.info("Connexion admin : %s", email_clean)
            return {"email": ADMIN_EMAIL, "is_admin": True}
        log.info("Mot de passe admin incorrect")
        return None

    # Utilisateurs normaux
    users = load_users()
    user = users.get(email_key)
    if not user:
        log.info("Utilisateur introuvable : %s", email_clean)
        return None

    stored_hash = user.get("password_hash", "")
    if _verify_password(password, stored_hash):
        log.info("Connexion utilisateur : %s", email_clean)
        return {"email": user["email"], "is_admin": False}

    log.info("Mot de passe incorrect pour %s", email_clean)
    return None


def is_admin(user: Optional[dict]) -> bool:
    """Retourne True si l'utilisateur est admin."""
    return bool(user and user.get("is_admin"))
