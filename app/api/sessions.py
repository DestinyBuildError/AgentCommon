"""会话 CRUD + 打断路由（契约）。薄协议转换，不含 Agent 逻辑。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_manager
from app.core.config import get_settings
from app.core.session_manager import SessionManager
from app.db.database import get_db
from app.db.repository import Repository
from app.schemas.session import SessionCreate, SessionOut
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(
    body: SessionCreate, db: AsyncSession = Depends(get_db)
) -> SessionOut:
    """创建会话。model / effort 缺省回落到全局 Settings。"""
    settings = get_settings()
    repo = Repository(db)
    session = await repo.create_session(
        model=body.model or settings.model,
        effort=body.effort or settings.effort,
        title=body.title,
        user_id=body.user_id,
    )
    return session


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[SessionOut]:
    """会话列表（H5 抽屉）：最近活跃在前。"""
    repo = Repository(db)
    sessions = await repo.list_sessions(limit=min(max(limit, 1), 500), offset=max(offset, 0))
    return list(sessions)


@router.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionOut:
    """会话详情 + 累计用量。"""
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> None:
    """删除会话：先停运行任务与事件订阅者，再级联删除消息/审计。"""
    repo = Repository(db)
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    await manager.stop_session(session_id)
    manager.bus.close_session(session_id)
    await repo.delete_session(session_id)


@router.post("/sessions/{session_id}/interrupt")
async def interrupt_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """打断当前轮。无运行中的 run 时 interrupted=false 仍返回 200。"""
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    interrupted = await manager.interrupt(session_id)
    return {"ok": True, "session_id": session_id, "interrupted": interrupted}
