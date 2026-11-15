"""任务拆解选择回调路由：POST /sessions/{id}/choices/{choice_id}。

与 approvals.py 同构的薄协议转换：校验会话存在 → manager.resolve_choice
设置 PendingChoice Future → AgentLoop 注入选择注记后继续循环。
无此 pending（已超时自动采用 / 会话不在跑）→ 404。
"""

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
    """提交任务拆解选择。无此 pending 选择（已超时/已决议）→ 404。"""
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
