
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.api.deps import get_workspace_service
from app.core.workspace_registry import WorkspaceError, WorkspaceRegistry
from app.schemas.workspace import WorkspaceCreateIn, WorkspaceDirOut

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def _error_response(exc: WorkspaceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": exc.code, "message": str(exc)}},
    )


@router.get("")
async def list_workspace_dirs(service: WorkspaceRegistry = Depends(get_workspace_service)) -> dict:
    dirs = [WorkspaceDirOut.model_validate(r).model_dump(mode="json") for r in service.snapshot()]
    return {"ok": True, "dirs": dirs}


@router.post("", status_code=201)
async def add_workspace_dir(
    body: WorkspaceCreateIn,
    service: WorkspaceRegistry = Depends(get_workspace_service),
) -> dict:
    try:
        record = await service.add(body.path, source="user")
    except WorkspaceError as exc:
        return _error_response(exc)
    return {"ok": True, "dir": WorkspaceDirOut.model_validate(record).model_dump(mode="json")}


@router.delete("/{dir_id}", status_code=204)
async def delete_workspace_dir(
    dir_id: str,
    service: WorkspaceRegistry = Depends(get_workspace_service),
) -> Response:
    try:
        await service.remove(dir_id)
    except WorkspaceError as exc:
        return _error_response(exc)
    return Response(status_code=204)
