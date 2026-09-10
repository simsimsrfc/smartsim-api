"""GET/POST /api/user/bankroll — gestion de la bankroll utilisateur."""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("SmartSim.api.bankroll")
router = APIRouter()

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
    except Exception:
        return None


def _resolve_user_email(x_user_email: Optional[str]) -> str:
    email = (x_user_email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "X-User-Email header requis")
    return email


class BankrollUpdate(BaseModel):
    amount: float = Field(..., ge=0)
    currency: str = Field("EUR", max_length=8)


@router.get("", summary="Bankroll actuelle de l'utilisateur")
@router.get("/", summary="Bankroll actuelle de l'utilisateur")
async def get_bankroll(x_user_email: Optional[str] = Header(default=None, alias="X-User-Email")):
    email = _resolve_user_email(x_user_email)
    client = _get_client()
    if not client:
        return {"amount": 0.0, "currency": "EUR", "user_email": email, "supabase": False}
    try:
        r = (client.table("user_bankroll").select("*")
                 .eq("user_email", email).limit(1).execute())
        if r.data:
            row = r.data[0]
            return {"amount": float(row.get("amount") or 0), "currency": row.get("currency") or "EUR",
                      "updated_at": row.get("updated_at"), "user_email": email}
        return {"amount": 0.0, "currency": "EUR", "user_email": email, "empty": True}
    except Exception as e:
        log.warning("get_bankroll %s : %s", email, e)
        raise HTTPException(500, "bankroll fetch failed")


@router.post("", summary="Met à jour la bankroll de l'utilisateur")
@router.post("/", summary="Met à jour la bankroll de l'utilisateur")
async def set_bankroll(payload: BankrollUpdate,
                          x_user_email: Optional[str] = Header(default=None, alias="X-User-Email")):
    email = _resolve_user_email(x_user_email)
    client = _get_client()
    if not client:
        raise HTTPException(503, "Supabase indisponible")
    try:
        client.table("user_bankroll").upsert({
            "user_email": email,
            "amount": float(payload.amount),
            "currency": (payload.currency or "EUR").upper()[:8],
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, on_conflict="user_email").execute()
        return {"ok": True, "amount": float(payload.amount), "currency": payload.currency}
    except Exception as e:
        log.warning("set_bankroll %s : %s", email, e)
        raise HTTPException(500, "bankroll update failed")


class BetLog(BaseModel):
    fixture_id: str
    market: str
    pick: str
    odd: float
    stake: float
    stake_pct: float
    proba: float


@router.post("/bets", summary="Enregistre un pari suivi")
async def log_bet(payload: BetLog,
                     x_user_email: Optional[str] = Header(default=None, alias="X-User-Email")):
    email = _resolve_user_email(x_user_email)
    client = _get_client()
    if not client:
        raise HTTPException(503, "Supabase indisponible")
    try:
        client.table("user_bets").insert({
            "user_email": email, "fixture_id": payload.fixture_id,
            "market": payload.market, "pick": payload.pick,
            "odd": float(payload.odd), "stake": float(payload.stake),
            "stake_pct": float(payload.stake_pct), "proba": float(payload.proba),
        }).execute()
        return {"ok": True}
    except Exception as e:
        log.warning("log_bet %s : %s", email, e)
        raise HTTPException(500, "bet log failed")


@router.get("/bets", summary="Historique des paris enregistrés")
async def list_bets(x_user_email: Optional[str] = Header(default=None, alias="X-User-Email")):
    email = _resolve_user_email(x_user_email)
    client = _get_client()
    if not client: return {"count": 0, "bets": []}
    try:
        r = (client.table("user_bets").select("*").eq("user_email", email)
                 .order("created_at", desc=True).limit(200).execute())
        bets = r.data or []
        return {"count": len(bets), "bets": bets}
    except Exception as e:
        log.warning("list_bets %s : %s", email, e)
        return {"count": 0, "bets": []}
