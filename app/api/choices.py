
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_manager
from app.core.session_manager import SessionManager
from app.db.database import get_db
from app.db.repository import Repository
from app.schemas.choice import ChoiceDecision
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["choices"])


@router.post("/sessions/{session_id}/choices/{choice_id}")
async def resolve_choice(
    session_id: str,
    choice_id: str,
    body: ChoiceDecision,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    resolved = await manager.resolve_choice(session_id, choice_id, body.selected)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"no pending choice for choice_id: {choice_id}",
        )
    return {
        "ok": True,
        "session_id": session_id,
        "choice_id": choice_id,
        "selected": body.selected,
    }
