"""FastAPI 入口。

启动（lifespan）：init_db 建表 → 对非 idle 状态会话跑 repair_orphans 并回写
（不变量 1：无孤儿 tool_use）→ 构建多目录工作区服务 WorkspaceRegistry
（DB 加载 + 空表播种）→ 构建 LLMClient / ToolRegistry /
Dispatcher 工厂 → 挂 SessionManager 到 app.state.sessions。
关闭：停全部会话任务 → dispose_engine。

全局异常处理器统一 JSON 错误格式：{"ok": false, "error": {"code", "message"}}。
测试注入 mock LLM：monkeypatch app.main.build_session_manager。
"""

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

# 工具沙箱默认根目录（read_file / search_workspace 的活动范围，安全设计）
# 仅作 WorkspaceRegistry 不可用时的兜底 provider；正常路径走 lifespan 构建的
# 多目录服务（provider 模式实时查询）。路径可为绝对或相对（CWD）
def _workspace_root() -> Path:
    try:
        from app.core.config import get_settings
        return Path(get_settings().workspace_root)
    except Exception:
        return Path("../workspace")

# lifespan 构建的多目录工作区服务（进程内单例；测试可经 lifespan 重置）。
# build_session_manager 的测试替身经 get_workspace_service() 定位同一实例，
# 与 /api/workspaces 端点共享同一份目录集合。
_workspace_service: WorkspaceRegistry | None = None


def get_workspace_service() -> WorkspaceRegistry | None:
    """当前进程的 WorkspaceRegistry（lifespan 构建后非 None）。"""
    return _workspace_service


# skills 根目录（渐进加载；可选——目录不存在时为空能力集）
SKILLS_ROOT = Path("./skills")

# 自扩展 Agent进程内单例：工具目录 / MCP 管理器 / 广播回调。
# 与 _workspace_service 同一模式——lifespan 先建好再构建 SessionManager；
# MCP 管理器构建晚于 SessionManager，故 meta 工具经 provider（执行时调
# get_mcp_manager()）迟绑定，注册期不依赖实例。
_tool_catalog: ToolCatalog | None = None
_mcp_manager_ref: MCPManager | None = None
_tools_broadcaster: Callable | None = None


def get_tool_catalog() -> ToolCatalog | None:
    """当前进程的工具目录（lifespan 构建后非 None）。"""
    return _tool_catalog


def get_mcp_manager() -> MCPManager | None:
    """当前进程的 MCPManager（lifespan 构建后非 None；meta 工具执行时取）。"""
    return _mcp_manager_ref


def get_tools_broadcaster() -> Callable | None:
    """install/uninstall 的 tools_changed 广播回调。"""
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
    """EventBus 的 DB 推导加载器：每次调用取最新 sessionmaker（测试可换引擎）。"""

    async def loader(session_id: str):
        factory = get_sessionmaker()
        async with factory() as db:
            return await Repository(db).get_messages(session_id)

    return loader


def _make_workspace_broadcaster(application: FastAPI):
    """workspace 变更 → 向所有运行中会话的 bus emit workspace_changed。

    全局事件模型：所有会话共享进程级 bus；按
    bus.known_session_ids() 广播——含发过消息的会话与**只挂了 SSE 流的
    冷会话**（冷会话此前收不到实时广播）。对无订阅者
    的会话只是入有界缓冲，该会话重连时 after_seq 重放照样补齐。漏事件
    仍可容忍——前端面板打开时会 GET /api/workspaces refetch 兜底。
    manager 在本回调绑定后才创建，故运行时经 app.state 迟绑定获取。
    """

    async def on_change(action, record, dirs) -> None:
        manager: SessionManager | None = getattr(application.state, "sessions", None)
        if manager is None:
            return  # 启动播种时尚无会话管理器
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
    """install/uninstall 成功 → 向所有已知会话广播 tools_changed。

    复用 workspace_changed 的全局广播模式：按 bus.known_session_ids() 发
    （含冷会话），统一信封 {seq, type, data} 由 bus.emit 编号；无订阅者时
    只是入有界缓冲，重连时 after_seq 重放补齐。前端据此 toast"工具集已更新"。
    """

    async def on_change(action: str, server: str, tools: list[str]) -> None:
        manager: SessionManager | None = getattr(application.state, "sessions", None)
        if manager is None:
            return  # 启动早期尚无会话管理器
        payload = ToolsChangedData(action=action, server=server, tools=list(tools))
        for session_id in manager.bus.known_session_ids():
            manager.bus.emit(session_id, SSEEventType.TOOLS_CHANGED, payload)

    return on_change


def build_registry() -> ToolRegistry:
    """本地工具 + 工作区管理工具 + load_skill+ 四个元工具（任务
    #11）；MCP 工具由 MCPManager 启动时追加注册（运行时安装亦走同一注册表）。

    workspace 工具走 provider 模式：roots_provider=服务.get_dirs（执行时
    实时查询当前目录集合，增删即刻生效，无重注册）；服务未构建时（理论
    不可达，lifespan 先建服务）回落 config workspace_root 单目录。
    元工具同走 provider 模式：mcp_provider=get_mcp_manager（执行时取
    MCPManager，规避构建顺序）；目录缺失时（理论不可达）不向模型暴露
    无法执行的工具。
    """
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
    """基础 system prompt + skills 一行描述段（稳定排序 → 可缓存）。"""
    skills = load_skills_from_dir(SKILLS_ROOT)
    return _SYSTEM_PROMPT + build_prompt_section(skills)


def build_anthropic_client(settings) -> AsyncAnthropic:
    """共享 AsyncAnthropic 客户端（LLM 调用与 compaction beta 共用连接池）。"""
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
    """上下文压缩器（anthropic beta compact-2026-01-12）。

    真实 API 联调列为遗留项：无真实 key 时仅在阈值触达时报错并被
    AgentLoop 降级（warning + 继续主循环），不影响其余功能。
    """
    return Compactor(
        client=build_anthropic_client(settings),
        model=settings.model,
    )


def build_mcp_configs() -> list[MCPServerConfig]:
    """MCP server 配置：Settings（env）+ config.yaml 覆盖（yaml 供结构化列表，
    Settings 里同名键兜底；mcp_enabled=false 可整体关闭，测试用）。"""
    settings = get_settings()
    if not settings.mcp_enabled:
        return []
    yaml_overrides = load_yaml_overrides(Path("config.yaml"))
    raw = yaml_overrides.get("mcp_servers", settings.mcp_servers)
    return parse_mcp_servers(raw)


def build_session_manager(bus: SessionEventBus) -> SessionManager:
    """组装引擎依赖（测试 monkeypatch 本函数注入 ScriptedLLM）。"""
    settings = get_settings()
    registry = build_registry()

    def dispatcher_factory(approve_callback):
        # dispatcher 每会话独立（AgentLoop.__init__ 会 set_approval_notifier，
        # 且熔断计数按会话隔离），registry 只读共享
        return ToolDispatcher(
            registry,
            approve_callback=approve_callback,
            breaker_limit=settings.tool_breaker_limit,
        )

    # 任务拆解规划器：与 LLM/compactor 共用同一 AsyncAnthropic
    # 连接池；planner_enabled=false → None，消息直接进循环
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
    # 启动恢复：非 idle 会话跑 repair_orphans 并回写，状态归位 idle
    repaired = await repair_orphan_sessions(get_sessionmaker())
    if repaired:
        logger.warning("startup repair_orphans fixed sessions: %s", repaired)
    # 多目录工作区服务：DB 加载 → 空表时用 config workspace_root
    # 播种（source=user）→ 先于 build_session_manager 挂好（注册工具要用）
    workspace_service = WorkspaceRegistry(
        get_sessionmaker(), on_change=_make_workspace_broadcaster(application)
    )
    await workspace_service.load()
    if await workspace_service.seed_if_empty(get_settings().workspace_root):
        logger.info("workspace_dirs was empty; seeded from config workspace_root")
    _workspace_service = workspace_service
    application.state.workspaces = workspace_service
    # 自扩展 Agent：工具目录 + 广播回调先于 build_session_manager
    # 挂好（register_meta_tools 注册时要用）；目录缺失降级为空目录不阻塞
    _tool_catalog = load_tool_catalog(get_settings().tool_catalog_path)
    logger.info("tool catalog loaded: %d server(s)", len(_tool_catalog.all()))
    _tools_broadcaster = _make_tools_broadcaster(application)
    bus = SessionEventBus(history_loader=_make_history_loader())
    manager = build_session_manager(bus)
    application.state.sessions = manager
    # 全局权限模式：从 app_settings 装载持久化值（无记录用默认
    # read 最保守）；此刻尚无运行会话，直接赋值即可
    async with get_sessionmaker()() as db:
        persisted = await Repository(db).get_setting("permission_mode")
    if persisted in ("read", "write", "full"):
        manager.permission_mode = persisted
        logger.info("permission mode loaded: %s", persisted)
    # MCP 接入（）：启动失败降级不阻塞——单 server 起不来只跳过
    mcp_manager = MCPManager(build_mcp_configs(), manager.registry)
    application.state.mcp = mcp_manager
    _mcp_manager_ref = mcp_manager  # meta 工具经 get_mcp_manager() 迟绑定
    connected = await mcp_manager.start()
    if connected:
        logger.info("MCP servers connected: %s", connected)
    try:
        yield
    finally:
        await manager.stop_all()
        await mcp_manager.stop()
        await dispose_engine()
        # 防跨测试/跨启动泄漏（下次 lifespan 重建）
        _workspace_service = None
        _tool_catalog = None
        _mcp_manager_ref = None
        _tools_broadcaster = None


def _error_payload(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="AgentCommon Backend",
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
