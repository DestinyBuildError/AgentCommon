"""审批回调路由（契约）：POST /sessions/{id}/approvals/{tool_use_id}。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_manager
from app.core.session_manager import SessionManager
from app.db.database import get_db
from app.db.repository import Repository
from app.schemas.approval import ApprovalDecision
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["approvals"])


@router.post("/sessions/{session_id}/approvals/{tool_use_id}")
async def resolve_approval(
    session_id: str,
    tool_use_id: str,
    body: ApprovalDecision,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """提交审批结果。无此 pending 审批（已决议 / 超时 / 会话不在跑）→ 404。"""
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    allowed = await manager.resolve_approval(
        session_id, tool_use_id, body.result == "allow"
    )
    if not allowed:
        raise HTTPException(
            status_code=404,
            detail=f"no pending approval for tool_use_id: {tool_use_id}",
        )
    return {
        "ok": True,
        "session_id": session_id,
        "tool_use_id": tool_use_id,
        "result": body.result,
    }
