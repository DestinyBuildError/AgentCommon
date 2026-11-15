"""API 路由层（薄，只做协议转换，不含 Agent 逻辑——分层职责）。

路由模块按 契约挂到 /api 前缀下：
sessions / messages / events / approvals / workspaces
（interrupt 路由在 sessions.py；共享依赖在 deps.py。）
"""

from fastapi import APIRouter

from app.api import approvals, choices, events, messages, permissions, sessions, workspaces

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions.router)
api_router.include_router(messages.router)
api_router.include_router(events.router)
api_router.include_router(approvals.router)
api_router.include_router(workspaces.router)
api_router.include_router(choices.router)
api_router.include_router(permissions.router)
