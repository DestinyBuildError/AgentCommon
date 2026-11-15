
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
    return {"ok": True, "mode": manager.permission_mode}


@router.put("/permissions")
async def update_permissions(
    body: PermissionUpdate,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    if body.mode not in PERMISSION_MODES:
        raise HTTPException(status_code=422, detail=f"invalid mode: {body.mode}")
    await manager.set_permission_mode(body.mode)
    return {"ok": True, "mode": manager.permission_mode}
