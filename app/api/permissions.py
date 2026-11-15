"""权限控制路由：全局权限模式的查询与切换。

GET /api/permissions → 当前模式（前端启动时拉取徽标状态）
PUT /api/permissions → 切换模式（写库持久化 + 传播到运行中的 dispatcher）
"""

from __future__ import annotations

from app.api.deps import get_session_manager
from app.core.dispatcher import PERMISSION_MODES
from app.core.session_manager import SessionManager
from app.schemas.permission import PermissionOut, PermissionUpdate
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["permissions"])


@router.get("/permissions")
async def get_permissions(
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """当前全局权限模式。"""
    return {"ok": True, "mode": manager.permission_mode}


@router.put("/permissions")
async def update_permissions(
    body: PermissionUpdate,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """切换全局权限模式：内存生效 + 持久化 + 传播到运行中的 dispatcher。"""
    if body.mode not in PERMISSION_MODES:
        # Literal 校验兜底（pydantic 已挡，这里防内部直调）
        raise HTTPException(status_code=422, detail=f"invalid mode: {body.mode}")
    await manager.set_permission_mode(body.mode)
    return {"ok": True, "mode": manager.permission_mode}
