
from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.session_manager import SessionManager
from app.core.workspace_registry import WorkspaceRegistry


def get_session_manager(request: Request) -> SessionManager:
    manager = getattr(request.app.state, "sessions", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="SessionManager is not ready")
    return manager


def get_workspace_service(request: Request) -> WorkspaceRegistry:
    service = getattr(request.app.state, "workspaces", None)
    if service is None:
        raise HTTPException(status_code=503, detail="WorkspaceRegistry is not ready")
    return service
