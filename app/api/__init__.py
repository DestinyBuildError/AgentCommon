
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
