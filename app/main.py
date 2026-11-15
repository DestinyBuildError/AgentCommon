
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.core.config import get_settings, load_yaml_overrides
from app.core.context import Compactor
from app.core.dispatcher import ToolDispatcher
from app.core.event_bus import SessionEventBus
from app.core.llm_client import SDK_MAX_RETRIES, LLMClient
from app.core.planner import TaskPlanner
from app.core.session_manager import SessionManager, repair_orphan_sessions
from app.core.workspace_registry import WorkspaceRegistry
from app.db.database import dispose_engine, get_sessionmaker, init_db
from app.db.repository import Repository
from app.mcp_loader import MCPManager, MCPServerConfig, parse_mcp_servers
from app.schemas.events import SSEEventType, ToolsChangedData
from app.schemas.workspace import WorkspaceChangedData, WorkspaceDirOut
from app.skills_loader import (
    build_prompt_section,
    load_skills_from_dir,
    register_load_skill_tool,
)
from app.tools.builtin import register_default_tools
from app.tools.catalog import ToolCatalog, load_tool_catalog
from app.tools.meta_tools import register_meta_tools
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

def _workspace_root() -> Path:
    try:
        from app.core.config import get_settings
        return Path(get_settings().workspace_root)
    except Exception:
        return Path("../workspace")

_workspace_service: WorkspaceRegistry | None = None


def get_workspace_service() -> WorkspaceRegistry | None:
    return _workspace_service


SKILLS_ROOT = Path("./skills")

_tool_catalog: ToolCatalog | None = None
_mcp_manager_ref: MCPManager | None = None
_tools_broadcaster: Callable | None = None


def get_tool_catalog() -> ToolCatalog | None:
    return _tool_catalog


def get_mcp_manager() -> MCPManager | None:
    return _mcp_manager_ref


def get_tools_broadcaster() -> Callable | None:
    return _tools_broadcaster

_SYSTEM_PROMPT = "You are a helpful agent embedded in an H5 chat app."

_HTTP_ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    503: "service_unavailable",
}


def _make_history_loader():

    async def loader(session_id: str):
        factory = get_sessionmaker()
        async with factory() as db:
            return await Repository(db).get_messages(session_id)

    return loader


def _make_workspace_broadcaster(application: FastAPI):

    async def on_change(action, record, dirs) -> None:
        manager: SessionManager | None = getattr(application.state, "sessions", None)
        if manager is None:
            return
        payload = WorkspaceChangedData(
            action=action,
            path=record.path,
            source=record.source,
            dirs=[WorkspaceDirOut.model_validate(d) for d in dirs],
        )
        for session_id in manager.bus.known_session_ids():
            manager.bus.emit(session_id, SSEEventType.WORKSPACE_CHANGED, payload)

    return on_change


def _make_tools_broadcaster(application: FastAPI):

    async def on_change(action: str, server: str, tools: list[str]) -> None:
        manager: SessionManager | None = getattr(application.state, "sessions", None)
        if manager is None:
            return
        payload = ToolsChangedData(action=action, server=server, tools=list(tools))
        for session_id in manager.bus.known_session_ids():
            manager.bus.emit(session_id, SSEEventType.TOOLS_CHANGED, payload)

    return on_change


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    service = _workspace_service
    if service is not None:
        register_default_tools(registry, service.get_dirs, workspace_service=service)
    else:
        register_default_tools(registry, _workspace_root())
    if _tool_catalog is not None:
        register_meta_tools(
            registry,
            _tool_catalog,
            mcp_provider=get_mcp_manager,
            on_change=_tools_broadcaster,
        )
    skills = load_skills_from_dir(SKILLS_ROOT)
    register_load_skill_tool(registry, skills)
    return registry


def build_system_prompt() -> str:
    skills = load_skills_from_dir(SKILLS_ROOT)
    return _SYSTEM_PROMPT + build_prompt_section(skills)


def build_anthropic_client(settings) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        max_retries=SDK_MAX_RETRIES,
    )


def build_llm() -> LLMClient:
    settings = get_settings()
    return LLMClient(
        client=build_anthropic_client(settings),
        model=settings.model,
        max_tokens=settings.max_tokens,
    )


def build_compactor(settings) -> Compactor:
    return Compactor(
        client=build_anthropic_client(settings),
        model=settings.model,
    )


def build_mcp_configs() -> list[MCPServerConfig]:
    settings = get_settings()
    if not settings.mcp_enabled:
        return []
    yaml_overrides = load_yaml_overrides(Path("config.yaml"))
    raw = yaml_overrides.get("mcp_servers", settings.mcp_servers)
    return parse_mcp_servers(raw)


def build_session_manager(bus: SessionEventBus) -> SessionManager:
    settings = get_settings()
    registry = build_registry()

    def dispatcher_factory(approve_callback):
        return ToolDispatcher(
            registry,
            approve_callback=approve_callback,
            breaker_limit=settings.tool_breaker_limit,
        )

    planner = None
    if settings.planner_enabled:
        planner = TaskPlanner(
            client=build_anthropic_client(settings),
            model=settings.model,
            max_options=settings.planner_max_options,
        )

    return SessionManager(
        session_factory=get_sessionmaker(),
        bus=bus,
        llm=build_llm(),
        registry=registry,
        dispatcher_factory=dispatcher_factory,
        system_prompt=build_system_prompt(),
        model=settings.model,
        max_iterations=settings.max_iterations,
        max_tokens=settings.max_tokens,
        approval_timeout_s=settings.approval_timeout_s,
        idle_timeout_s=settings.session_idle_timeout_s,
        compactor=build_compactor(settings),
        planner=planner,
        choice_timeout_s=settings.planner_timeout_s,
        progress_interval_s=settings.progress_interval_s,
        long_running_after_s=settings.long_running_after_s,
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _workspace_service, _tool_catalog, _mcp_manager_ref, _tools_broadcaster
    await init_db()
    repaired = await repair_orphan_sessions(get_sessionmaker())
    if repaired:
        logger.warning("startup repair_orphans fixed sessions: %s", repaired)
    workspace_service = WorkspaceRegistry(
        get_sessionmaker(), on_change=_make_workspace_broadcaster(application)
    )
    await workspace_service.load()
    if await workspace_service.seed_if_empty(get_settings().workspace_root):
        logger.info("workspace_dirs was empty; seeded from config workspace_root")
    _workspace_service = workspace_service
    application.state.workspaces = workspace_service
    _tool_catalog = load_tool_catalog(get_settings().tool_catalog_path)
    logger.info("tool catalog loaded: %d server(s)", len(_tool_catalog.all()))
    _tools_broadcaster = _make_tools_broadcaster(application)
    bus = SessionEventBus(history_loader=_make_history_loader())
    manager = build_session_manager(bus)
    application.state.sessions = manager
    async with get_sessionmaker()() as db:
        persisted = await Repository(db).get_setting("permission_mode")
    if persisted in ("read", "write", "full"):
        manager.permission_mode = persisted
        logger.info("permission mode loaded: %s", persisted)
    mcp_manager = MCPManager(build_mcp_configs(), manager.registry)
    application.state.mcp = mcp_manager
    _mcp_manager_ref = mcp_manager
    connected = await mcp_manager.start()
    if connected:
        logger.info("MCP servers connected: %s", connected)
    try:
        yield
    finally:
        await manager.stop_all()
        await mcp_manager.stop()
        await dispose_engine()
        _workspace_service = None
        _tool_catalog = None
        _mcp_manager_ref = None
        _tools_broadcaster = None


def _error_payload(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="LLM Agent Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(api_router)

    @application.get("/health", tags=["meta"])
    async def health() -> dict:
        return {
            "status": "ok",
            "version": application.version,
            "model": settings.model,
            "effort": settings.effort,
        }

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _HTTP_ERROR_CODES.get(exc.status_code, f"http_{exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, str(exc.detail)),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        summary = "; ".join(
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}"
            for err in exc.errors()[:5]
        )
        return JSONResponse(
            status_code=422,
            content=_error_payload("validation_error", summary or "invalid request"),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unhandled error on %s", _request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", str(exc)),
        )

    return application


app = create_app()
