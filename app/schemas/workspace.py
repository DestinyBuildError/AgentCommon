"""工作区目录 API 模型（多目录工作区的前后端契约）。

GET /api/workspaces → {"ok": true, "dirs": [WorkspaceDirOut...]}
POST /api/workspaces {"path": ...} → 201 {"ok": true, "dir": WorkspaceDirOut}
DELETE /api/workspaces/{id} → 204
SSE workspace_changed → WorkspaceChangedData（广播载荷）
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class WorkspaceDirOut(BaseModel):
    """单个工作区目录（ORM WorkspaceDir 的对外形态）。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    path: str
    source: Literal["user", "agent"]
    created_at: datetime


class WorkspaceListOut(BaseModel):
    """GET /api/workspaces 响应体。"""

    ok: bool = True
    dirs: list[WorkspaceDirOut]


class WorkspaceCreateIn(BaseModel):
    """POST /api/workspaces 请求体。path 必须是已存在的绝对目录路径。"""

    path: str


class WorkspaceDirCreatedOut(BaseModel):
    """POST /api/workspaces 响应体（201）。"""

    ok: bool = True
    dir: WorkspaceDirOut


class WorkspaceChangedData(BaseModel):
    """event: workspace_changed —— 工作区变更广播（全局事件）。

    面板打开时可用 GET /api/workspaces refetch 兜底；漏事件可容忍。
    dirs 携带变更后的完整当前列表，前端可直接整体替换。
    """

    action: Literal["add", "remove"]
    path: str
    source: Literal["user", "agent"]
    dirs: list[WorkspaceDirOut]
