"""API 层共享依赖（避免路由模块与 api 包 __init__ 循环导入）。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.session_manager import SessionManager
from app.core.workspace_registry import WorkspaceRegistry


def get_session_manager(request: Request) -> SessionManager:
    """从 app.state 取 lifespan 构建的 SessionManager。"""
    manager = getattr(request.app.state, "sessions", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="SessionManager is not ready")
    return manager


def get_workspace_service(request: Request) -> WorkspaceRegistry:
    """从 app.state 取 lifespan 构建的 WorkspaceRegistry（多目录工作区服务）。"""
    service = getattr(request.app.state, "workspaces", None)
    if service is None:
        raise HTTPException(status_code=503, detail="WorkspaceRegistry is not ready")
    return service
