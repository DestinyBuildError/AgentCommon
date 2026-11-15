
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.session_manager import SessionManager
from app.db.database import get_db
from app.db.repository import Repository
from app.schemas.message import MessageCreate, MessageOut
from app.api.deps import get_session_manager
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["messages"])


@router.post("/sessions/{session_id}/messages", status_code=202)
async def post_message(
    session_id: str,
    body: MessageCreate,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    repo = Repository(db)
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    manager.enqueue_message(session_id, body.text)
    return {"ok": True, "session_id": session_id, "queued": True}


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def get_messages(
    session_id: str,
    after_seq: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    repo = Repository(db)
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    messages = await repo.get_messages(session_id, after_seq=max(after_seq, 0))
    return list(messages)
