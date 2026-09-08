"""
Routes /api/favorites/*  — toutes protégées par auth Supabase.

GET    /api/favorites              — Lister les favoris du user (matchs + équipes)
POST   /api/favorites/toggle       — Ajouter ou retirer (idempotent)
DELETE /api/favorites/{id}         — Supprimer un favori par son id PK
"""

import logging
import traceback
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_current_user, get_supabase

log = logging.getLogger("SmartSim.api.favorites")
router = APIRouter()


def _explain(e: Exception) -> str:
    """Extrait un message d'erreur lisible pour le client (sans secrets)."""
    # supabase-py wrappe souvent les erreurs PostgREST dans une APIError
    for attr in ("message", "detail", "code", "hint"):
        v = getattr(e, attr, None)
        if v:
            return f"{type(e).__name__}: {v}"
    return f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────
# Schémas Pydantic
# ──────────────────────────────────────────────
class ToggleBody(BaseModel):
    fav_type: Literal["match", "team"] = Field(description="Type de favori")
    target_id: str = Field(min_length=1, description="fixture_id ou team_id")
    label: Optional[str] = Field(default="", description="Affichage : nom équipe ou matchup")
    league_name: Optional[str] = Field(default="", description="Pour le rendu UI")
    extra_data: Optional[dict] = Field(default=None, description="Métadonnées libres (logo, date...)")


# ══════════════════════════════════════════════════════════════
# GET /api/favorites
# ══════════════════════════════════════════════════════════════
@router.get("", summary="Lister les favoris de l'utilisateur courant")
async def list_favorites(
    fav_type: Optional[Literal["match", "team"]] = None,
    user: dict = Depends(get_current_user),
):
    """
    Retourne les favoris du user. Filtrable par type (match | team).
    """
    client = get_supabase()
    try:
        q = client.table("favorites").select("*").eq("user_id", user["id"])
        if fav_type:
            q = q.eq("fav_type", fav_type)
        resp = q.order("created_at", desc=True).execute()
    except Exception as e:
        log.error("Erreur lecture favorites : %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Lecture favoris échouée — {_explain(e)}")

    rows = resp.data or []
    return {
        "count": len(rows),
        "items": rows,
    }


# ══════════════════════════════════════════════════════════════
# GET /api/favorites/_debug  — Diagnostic (à retirer après debug)
# ══════════════════════════════════════════════════════════════
@router.get("/_debug", summary="Diagnostic table favorites (dev only)")
async def debug_favorites(user: dict = Depends(get_current_user)):
    """
    Vérifie l'accès à la table `favorites` et retourne des infos de diagnostic.
    À retirer une fois le bug corrigé.
    """
    client = get_supabase()
    diagnostics = {
        "user_id_received": user["id"],
        "user_id_type": type(user["id"]).__name__,
        "user_email": user.get("email", ""),
    }

    # Test 1 : SELECT sans filtre (table accessible ?)
    try:
        r = client.table("favorites").select("id", count="exact").limit(1).execute()
        diagnostics["table_accessible"] = True
        diagnostics["total_rows_in_table"] = getattr(r, "count", None)
    except Exception as e:
        diagnostics["table_accessible"] = False
        diagnostics["table_error"] = _explain(e)
        return diagnostics

    # Test 2 : SELECT avec filtre user_id (le test qui échoue normalement)
    try:
        r = client.table("favorites").select("*").eq("user_id", user["id"]).execute()
        diagnostics["filter_by_user_ok"] = True
        diagnostics["user_rows"] = len(r.data or [])
    except Exception as e:
        diagnostics["filter_by_user_ok"] = False
        diagnostics["filter_error"] = _explain(e)

    # Test 3 : SELECT avec order
    try:
        r = client.table("favorites").select("*").eq("user_id", user["id"]).order("created_at", desc=True).execute()
        diagnostics["order_by_ok"] = True
    except Exception as e:
        diagnostics["order_by_ok"] = False
        diagnostics["order_error"] = _explain(e)

    return diagnostics


# ══════════════════════════════════════════════════════════════
# POST /api/favorites/toggle
# ══════════════════════════════════════════════════════════════
@router.post("/toggle", summary="Ajouter ou retirer un favori (idempotent)")
async def toggle_favorite(
    body: ToggleBody,
    user: dict = Depends(get_current_user),
):
    """
    Si l'élément (user_id, fav_type, target_id) existe déjà → suppression.
    Sinon → création.

    Retourne `{action: "added"|"removed", item: ...}`.
    """
    client = get_supabase()

    # 1. Existe déjà ?
    try:
        existing = (
            client.table("favorites")
            .select("id")
            .eq("user_id", user["id"])
            .eq("fav_type", body.fav_type)
            .eq("target_id", body.target_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        log.error("Erreur lookup favorites : %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Lookup favori échoué — {_explain(e)}")

    rows = existing.data or []
    if rows:
        # → DELETE
        fav_id = rows[0]["id"]
        try:
            client.table("favorites").delete().eq("id", fav_id).execute()
        except Exception as e:
            log.error("Erreur delete favori : %s\n%s", e, traceback.format_exc())
            raise HTTPException(500, f"Suppression échouée — {_explain(e)}")
        return {"action": "removed", "id": fav_id}

    # → INSERT
    payload = {
        "user_id": user["id"],
        "fav_type": body.fav_type,
        "target_id": body.target_id,
        "label": body.label or "",
        "league_name": body.league_name or "",
        "extra_data": body.extra_data,
    }
    try:
        ins = client.table("favorites").insert(payload).execute()
    except Exception as e:
        log.error("Erreur insert favori : %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Création échouée — {_explain(e)}")

    inserted = (ins.data or [None])[0]
    return {"action": "added", "item": inserted}


# ══════════════════════════════════════════════════════════════
# DELETE /api/favorites/{fav_id}
# ══════════════════════════════════════════════════════════════
@router.delete("/{fav_id}", summary="Supprimer un favori par son id")
async def delete_favorite(
    fav_id: int,
    user: dict = Depends(get_current_user),
):
    """Supprime un favori. Vérifie la propriété (user_id) pour la sécurité."""
    client = get_supabase()
    try:
        # Vérifier ownership
        resp = (
            client.table("favorites")
            .select("id, user_id")
            .eq("id", fav_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(500, f"Erreur : {e}")

    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Favori introuvable.")

    if str(rows[0]["user_id"]) != user["id"]:
        raise HTTPException(403, "Tu n'es pas propriétaire de ce favori.")

    try:
        client.table("favorites").delete().eq("id", fav_id).execute()
    except Exception as e:
        log.error("Erreur delete : %s", e)
        raise HTTPException(500, "Suppression échouée.")

    return {"status": "ok", "id": fav_id}
