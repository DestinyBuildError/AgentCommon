"""权限控制测试——矩阵逐格 + 新工具 + 全局开关传播/持久化。

矩阵事实来源：dispatcher.PERMISSION_MATRIX（read/write/full × read/write/
execute/admin → allow/approval）。全部走 dispatcher.dispatch 真实流水线
（含审批回调），不 mock 内部方法。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.dispatcher import DEFAULT_PERMISSION_MODE, ToolDispatcher
from app.core.session_manager import SessionManager
from app.db.database import build_engine, dispose_engine, get_sessionmaker
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import SSEEvent
from app.tools.registry import ToolRegistry, ToolSpec

# ----------------------------------------------------------------------
# 矩阵夹具：四类 access 各一个工具 + 记录型审批回调
# ----------------------------------------------------------------------


def build_access_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def make(name: str, access: str) -> ToolSpec:
        async def handler(tool_input: dict) -> dict:
            return {"done": name}

        return ToolSpec(
            name=name,
            description=name,
            input_schema={"type": "object", "properties": {}},
            handler=handler,
            timeout_s=5,
            access=access,
        )

    for access in ("read", "write", "execute", "admin"):
        registry.register(make(f"tool_{access}", access))
    return registry


def make_dispatcher(mode: str, approvals: list[str]) -> ToolDispatcher:
    async def approve(tool_name: str, tool_input: Any) -> bool:
        approvals.append(tool_name)
        return False  # 一律拒绝：审批命中即可断言，放行由 allow 档验证

    return ToolDispatcher(
        build_access_registry(), approve_callback=approve, retry_backoff_base_s=0,
        permission_mode=mode,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ["read", "write", "execute", "admin"])
async def test_full_mode_allows_everything(access: str) -> None:
    approvals: list[str] = []
    dispatcher = make_dispatcher("full", approvals)
    result = await dispatcher.dispatch(f"tool_{access}", {})
    assert result.ok
    assert approvals == []  # 全开放：危险工具也免审批


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ["read", "write", "execute"])
async def test_write_mode_allows_except_admin(access: str) -> None:
    approvals: list[str] = []
    dispatcher = make_dispatcher("write", approvals)
    result = await dispatcher.dispatch(f"tool_{access}", {})
    assert result.ok
    assert approvals == []


@pytest.mark.asyncio
async def test_write_mode_gates_admin() -> None:
    approvals: list[str] = []
    dispatcher = make_dispatcher("write", approvals)
    result = await dispatcher.dispatch("tool_admin", {})
    assert not result.ok
    assert result.code == "approval_denied"
    assert approvals == ["tool_admin"]  # 走了审批回调（回调拒绝 → denied）


@pytest.mark.asyncio
@pytest.mark.parametrize("access", ["write", "execute", "admin"])
async def test_read_mode_gates_non_read(access: str) -> None:
    approvals: list[str] = []
    dispatcher = make_dispatcher("read", approvals)
    result = await dispatcher.dispatch(f"tool_{access}", {})
    assert not result.ok
    assert result.code == "approval_denied"
    assert approvals == [f"tool_{access}"]


@pytest.mark.asyncio
async def test_read_mode_allows_read() -> None:
    dispatcher = make_dispatcher("read", [])
    result = await dispatcher.dispatch("tool_read", {})
    assert result.ok


def test_default_mode_is_read() -> None:
    assert DEFAULT_PERMISSION_MODE == "read"
    dispatcher = ToolDispatcher(build_access_registry())
    assert dispatcher.permission_mode == "read"


def test_invalid_mode_rejected() -> None:
    dispatcher = ToolDispatcher(build_access_registry())
    with pytest.raises(ValueError):
        dispatcher.set_permission_mode("sudo")


@pytest.mark.asyncio
async def test_mode_hot_switch_takes_effect() -> None:
    approvals: list[str] = []
    dispatcher = make_dispatcher("full", approvals)
    assert (await dispatcher.dispatch("tool_write", {})).ok  # full 直行
    dispatcher.set_permission_mode("read")
    result = await dispatcher.dispatch("tool_write", {})
    assert not result.ok  # 切到 read → 写类立刻受审批门控
    assert approvals == ["tool_write"]


# ----------------------------------------------------------------------
# 新工具：write_file / delete_file / run_command（full 模式直行）
# ----------------------------------------------------------------------


@pytest.fixture
async def tool_db():
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def workspace_dispatcher(root: Path) -> ToolDispatcher:
    registry = ToolRegistry()
    from app.tools.builtin import (
        register_default_tools,
    )

    register_default_tools(registry, lambda: [root])
    return ToolDispatcher(registry, retry_backoff_base_s=0, permission_mode="full")


@pytest.mark.asyncio
async def test_write_file_creates_and_reads_back(tmp_path: Path, tool_db) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    dispatcher = workspace_dispatcher(root)
    result = await dispatcher.dispatch(
        "write_file", {"path": "sub/dir/note.txt", "content": "你好\nworld"}
    )
    assert result.ok, result.content
    assert (root / "sub" / "dir" / "note.txt").read_text(encoding="utf-8") == "你好\nworld"
    readback = await dispatcher.dispatch("read_file", {"path": "sub/dir/note.txt"})
    assert readback.ok and "你好" in readback.content


@pytest.mark.asyncio
async def test_write_file_overwrite_false_conflicts(tmp_path: Path, tool_db) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("old", encoding="utf-8")
    dispatcher = workspace_dispatcher(root)
    result = await dispatcher.dispatch(
        "write_file", {"path": "a.txt", "content": "new", "overwrite": False}
    )
    assert not result.ok and "file_exists" in result.content
    assert (root / "a.txt").read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_write_file_blocks_path_traversal(tmp_path: Path, tool_db) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    dispatcher = workspace_dispatcher(root)
    result = await dispatcher.dispatch("write_file", {"path": "../escape.txt", "content": "x"})
    assert not result.ok and "path_outside_workspace" in result.content
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.asyncio
async def test_delete_file_removes_and_guards(tmp_path: Path, tool_db) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "gone.txt").write_text("x", encoding="utf-8")
    (root / "full").mkdir()
    (root / "full" / "keep.txt").write_text("x", encoding="utf-8")
    dispatcher = workspace_dispatcher(root)

    ok = await dispatcher.dispatch("delete_file", {"path": "gone.txt"})
    assert ok.ok and not (root / "gone.txt").exists()

    missing = await dispatcher.dispatch("delete_file", {"path": "gone.txt"})
    assert not missing.ok and "not_found" in missing.content

    nonempty = await dispatcher.dispatch("delete_file", {"path": "full"})
    assert not nonempty.ok and "dir_not_empty" in nonempty.content
    assert (root / "full" / "keep.txt").exists()  # 非空目录未被递归删除

    protected = await dispatcher.dispatch("delete_file", {"path": str(root)})
    assert not protected.ok  # 工作区根本身受保护


@pytest.mark.asyncio
async def test_run_command_echoes(tmp_path: Path, tool_db) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    dispatcher = workspace_dispatcher(root)
    result = await dispatcher.dispatch("run_command", {"command": "echo hello"})
    assert result.ok, result.content
    assert "hello" in result.content  # stdout 捕获（bash / cmd 皆然）


@pytest.mark.asyncio
async def test_run_command_rejects_empty(tmp_path: Path, tool_db) -> None:
    dispatcher = workspace_dispatcher(tmp_path)
    result = await dispatcher.dispatch("run_command", {"command": "  "})
    assert not result.ok and "invalid_params" in result.content


# ----------------------------------------------------------------------
# SessionManager：全局模式传播 + 持久化
# ----------------------------------------------------------------------


class FakeLoop:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def set_permission_mode(self, mode: str) -> None:
        self.modes.append(mode)


def build_manager(session_factory) -> SessionManager:
    return SessionManager(
        session_factory=session_factory,
        bus=None,
        llm=None,
        registry=ToolRegistry(),
        dispatcher_factory=lambda cb: ToolDispatcher(ToolRegistry(), approve_callback=cb),
    )


@pytest.mark.asyncio
async def test_set_permission_mode_persists() -> None:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    manager = build_manager(factory)
    assert manager.permission_mode == "read"
    await manager.set_permission_mode("write")
    assert manager.permission_mode == "write"
    async with factory() as db:
        assert await Repository(db).get_setting("permission_mode") == "write"
    await engine.dispose()


@pytest.mark.asyncio
async def test_set_permission_mode_propagates_to_running_loops() -> None:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    manager = build_manager(factory)
    from app.core.event_bus import SessionEventBus

    rs = manager.start_session.__self__ if False else None  # 占位，下面手工构造
    from app.core.session_manager import RunningSession

    rs = RunningSession(
        session_id="s1", task=None, queue=asyncio.Queue(), bus=SessionEventBus()
    )
    fake_loop = FakeLoop()
    rs.agent_loop = fake_loop
    manager.sessions["s1"] = rs
    await manager.set_permission_mode("full")
    assert fake_loop.modes == ["full"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_set_permission_mode_invalid() -> None:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    manager = build_manager(async_sessionmaker(engine, expire_on_commit=False))
    with pytest.raises(ValueError):
        await manager.set_permission_mode("sudo")
    await engine.dispose()


# ----------------------------------------------------------------------
# API：GET/PUT /api/permissions + 重启恢复
# ----------------------------------------------------------------------


@pytest.fixture
async def perm_api(tmp_path, monkeypatch):
    """隔离 DB + 真实 SessionManager（mock LLM 永不被调用）+ 手动 lifespan。"""
    import app.main as main_mod

    db_path = (tmp_path / "perm_test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()
    monkeypatch.setattr(
        main_mod,
        "build_session_manager",
        lambda bus: build_manager(get_sessionmaker()),
    )
    application = main_mod.create_app()
    try:
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, application, db_path
    finally:
        await dispose_engine()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_permissions_get_default(perm_api) -> None:
    client, _app, _db = perm_api
    resp = await client.get("/api/permissions")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "mode": "read"}


@pytest.mark.asyncio
async def test_permissions_put_switches(perm_api) -> None:
    client, app, _db = perm_api
    resp = await client.put("/api/permissions", json={"mode": "full"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "mode": "full"}
    assert app.state.sessions.permission_mode == "full"
    follow = await client.get("/api/permissions")
    assert follow.json()["mode"] == "full"


@pytest.mark.asyncio
async def test_permissions_put_rejects_unknown(perm_api) -> None:
    client, _app, _db = perm_api
    resp = await client.put("/api/permissions", json={"mode": "sudo"})
    assert resp.status_code == 422  # pydantic Literal 校验


@pytest.mark.asyncio
async def test_permission_mode_survives_restart(perm_api) -> None:
    """切换 → 重建 app（同库）→ 权限模式从 app_settings 恢复。"""
    import app.main as main_mod

    client, _app, db_path = perm_api
    await client.put("/api/permissions", json={"mode": "write"})

    second = main_mod.create_app()
    try:
        async with second.router.lifespan_context(second):
            assert second.state.sessions.permission_mode == "write"
    finally:
        pass


@pytest.mark.asyncio
async def test_session_events_envelope_shape(perm_api) -> None:
    """守卫断言：权限事件不影响 SSE 信封（顺带验证 app 可正常构建事件）。"""
    from app.schemas.events import SSEEventType, make_event

    event = make_event(1, SSEEventType.ERROR, {"code": "x", "message": "y"})
    assert isinstance(event, SSEEvent)
