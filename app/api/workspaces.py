"""工作区目录管理路由（多目录工作区的前后端契约）。薄协议转换。

GET /api/workspaces → 200 {"ok": true, "dirs": [...]}
POST /api/workspaces → 201 {"ok": true, "dir": {...}} | 400 invalid_path | 409 duplicate_path
DELETE /api/workspaces/{id} → 204 | 400 last_directory | 404 not_found

错误统一 {"ok": false, "error": {"code", "message"}}（领域错误码由
WorkspaceError.code 决定，不走全局 HTTP→code 映射）。变更成功后由
WorkspaceRegistry 的 on_change 回调广播 workspace_changed 到全部运行中
会话（本层不再另行广播）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response  # noqa: F401 (Response 用于 204)

from app.api.deps import get_workspace_service
from app.core.workspace_registry import WorkspaceError, WorkspaceRegistry
from app.schemas.workspace import WorkspaceCreateIn, WorkspaceDirOut

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def _error_response(exc: WorkspaceError) -> JSONResponse:
    """领域错误 → 统一 JSON 错误体（code/message 严格按契约）。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": exc.code, "message": str(exc)}},
    )


@router.get("")
async def list_workspace_dirs(service: WorkspaceRegistry = Depends(get_workspace_service)) -> dict:
    """当前工作区目录列表（启动播种 + 运行期增删后的完整集合）。"""
    dirs = [WorkspaceDirOut.model_validate(r).model_dump(mode="json") for r in service.snapshot()]
    return {"ok": True, "dirs": dirs}


@router.post("", status_code=201)
async def add_workspace_dir(
    body: WorkspaceCreateIn,
    service: WorkspaceRegistry = Depends(get_workspace_service),
) -> dict:
    """新增工作区目录（source=user）。校验：绝对路径 + 存在且为目录 +
    normcase 去重；成功即广播 workspace_changed。"""
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
    """删除工作区目录（按 id）。最后一个目录拒绝删除（last_directory）。"""
    try:
        await service.remove(dir_id)
    except WorkspaceError as exc:
        return _error_response(exc)
    return Response(status_code=204)
