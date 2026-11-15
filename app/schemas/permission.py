"""权限控制契约：全局权限模式的查询与切换。"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel

# 三档权限模式（语义见 dispatcher.PERMISSION_MATRIX）：
# read : 只读直行；文件写删 / 命令执行 / MCP 元操作一律审批
# write : 读 + 文件写删 + 命令执行直行；仅 MCP 元操作审批
# full : 全部直行（含危险工具免审批）
PermissionMode = Literal["read", "write", "full"]


class PermissionUpdate(BaseModel):
    """PUT /api/permissions 请求体：{mode: read|write|full}。"""

    mode: PermissionMode


class PermissionOut(BaseModel):
    """GET /api/permissions 响应体。"""

    mode: PermissionMode
