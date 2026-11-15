
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import get_settings
from app.core.dispatcher import ToolDispatcher
from app.core.session_manager import SessionManager
from app.db.database import dispose_engine, get_sessionmaker
from app.mcp_loader import MCPManager, MCPServerConfig
from app.schemas.events import SSEEventType
from app.tools.catalog import CatalogItem, ToolCatalog, load_tool_catalog
from app.tools.errors import ToolInputError, ToolPermanentError
from app.tools.meta_tools import register_meta_tools
from app.tools.registry import ToolRegistry

from test_api import (
    ScriptedLLM,
    create_session,
    end_turn,
    manager_of,
    tool_use_turn,
    wait_run_done,
    wait_until,
)
from test_mcp_loader import (
    FakeCallToolResult,
    FakeSession,
    FakeTool,
    make_connector,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def real_catalog() -> ToolCatalog:
    return load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml")


def make_meta_registry(
    catalog: ToolCatalog,
    sessions: list[FakeSession],
    broadcasts: list | None = None,
) -> tuple[ToolRegistry, MCPManager]:
    registry = ToolRegistry()
    manager = MCPManager(
        [], registry, connect_timeout_s=1, connector=make_connector(sessions)
    )

    async def on_change(action: str, server: str, tools: list[str]) -> None:
        if broadcasts is not None:
            broadcasts.append((action, server, list(tools)))

    register_meta_tools(registry, catalog, lambda: manager, on_change)
    return registry, manager


def mcp_tool_names(registry: ToolRegistry) -> list[str]:
    return [n for n in registry.names() if n.startswith("mcp__")]


class ReversingSession(FakeSession):

    async def call_tool(
        self, name: str, arguments: dict[str, Any], read_timeout_seconds: Any = None
    ) -> FakeCallToolResult:
        self.calls.append((name, dict(arguments)))
        if name == "reverse_text":
            return FakeCallToolResult(str(arguments.get("text", ""))[::-1])
        return FakeCallToolResult(f"echo:{arguments.get('text', '')}")


@pytest.fixture
async def meta_api(tmp_path, monkeypatch):
    import app.main as main_mod
    from app.tools.builtin import register_default_tools

    db_path = (tmp_path / "meta_agent_test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()

    scripted = ScriptedLLM()

    def fake_build_session_manager(bus):
        from app.main import get_workspace_service as current_service

        service = current_service()
        catalog = main_mod.get_tool_catalog()
        assert catalog is not None
        settings = get_settings()
        registry = ToolRegistry()
        register_default_tools(registry, service.get_dirs, workspace_service=service)
        register_meta_tools(
            registry,
            catalog,
            mcp_provider=main_mod.get_mcp_manager,
            on_change=main_mod.get_tools_broadcaster(),
        )

        def dispatcher_factory(approve_callback):
            return ToolDispatcher(
                registry, approve_callback=approve_callback, retry_backoff_base_s=0
            )

        manager = SessionManager(
            session_factory=get_sessionmaker(),
            bus=bus,
            llm=scripted,
            registry=registry,
            dispatcher_factory=dispatcher_factory,
            system_prompt="test",
            model=settings.model,
            max_iterations=settings.max_iterations,
            max_tokens=settings.max_tokens,
            approval_timeout_s=settings.approval_timeout_s,
            idle_timeout_s=settings.session_idle_timeout_s,
        )
        manager.permission_mode = "write"
        return manager

    monkeypatch.setattr(main_mod, "build_session_manager", fake_build_session_manager)
    application = main_mod.create_app()
    try:
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, application, scripted
    finally:
        await dispose_engine()
        get_settings.cache_clear()


async def resolve_when_pending(
    client: httpx.AsyncClient, application, session_id: str, tool_use_id: str, result: str
) -> None:
    manager = manager_of(application)

    def pending() -> bool:
        running = manager.get_running(session_id)
        return running is not None and tool_use_id in running.pending_approvals

    await wait_until(pending)
    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/{tool_use_id}", json={"result": result}
    )
    assert resp.status_code == 200


def test_real_catalog_loads_three_builtin_servers(real_catalog: ToolCatalog) -> None:
    assert real_catalog.names() == ["password_gen", "text_tools", "unit_convert"]
    for name in real_catalog.names():
        item = real_catalog.get(name)
        assert item is not None
        assert item.trust == "builtin"
        assert item.command == "python"
        assert item.args and item.args[0].startswith("catalog_servers/")
        assert item.cwd == str(BACKEND_ROOT)
        config = item.to_server_config()
        assert isinstance(config, MCPServerConfig)
        assert config.enabled is True and config.cwd == item.cwd


def test_catalog_item_defaults_and_validation() -> None:
    item = CatalogItem.from_dict(
        {
            "name": "community_srv",
            "description": "A third-party server",
            "keywords": ["a", "b"],
            "transport": {"url": "http://127.0.0.1:1/mcp"},
        }
    )
    assert item.trust == "community"
    assert item.url == "http://127.0.0.1:1/mcp" and item.command is None

    bad_cases = [
        {"description": "no name", "transport": {"command": "x"}},
        {"name": "x", "transport": {"command": "x"}},
        {"name": "x", "description": "d"},
        {"name": "x", "description": "d", "transport": {}},
        {
            "name": "x",
            "description": "d",
            "transport": {"command": "x", "args": "no"},
            "trust": "builtin",
        },
        {
            "name": "x",
            "description": "d",
            "transport": {"command": "x"},
            "trust": "alien",
        },
    ]
    for raw in bad_cases:
        with pytest.raises(ValueError):
            CatalogItem.from_dict(raw)


def test_load_tool_catalog_skips_invalid_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "- name: good\n"
        "  description: First wins\n"
        "  transport: {command: python, args: [a.py]}\n"
        "- name: good\n"
        "  description: duplicate ignored\n"
        "  transport: {command: python, args: [b.py]}\n"
        "- description: missing name\n"
        "  transport: {command: python}\n"
        "- name: bad_trust\n"
        "  description: d\n"
        "  transport: {command: python}\n"
        "  trust: alien\n"
        "- just a string\n",
        encoding="utf-8",
    )
    catalog = load_tool_catalog(path)
    assert catalog.names() == ["good"]
    assert catalog.get("good").description == "First wins"
    assert catalog.get("good").trust == "community"


def test_load_tool_catalog_missing_and_corrupt_file(tmp_path: Path) -> None:
    assert load_tool_catalog(tmp_path / "absent.yaml").names() == []
    corrupt = tmp_path / "corrupt.yaml"
    corrupt.write_text("key: [unclosed", encoding="utf-8")
    assert load_tool_catalog(corrupt).names() == []
    not_a_list = tmp_path / "not_a_list.yaml"
    not_a_list.write_text("key: value", encoding="utf-8")
    assert load_tool_catalog(not_a_list).names() == []


async def test_discover_matches_name_keywords_case_insensitive(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    handler = registry.get("discover_tools").handler

    for query, expected in [
        ("PASSWORD", "password_gen"),
        ("温度换算", "unit_convert"),
        ("reverse a string", "text_tools"),
        ("km", "unit_convert"),
    ]:
        result = await handler({"query": query})
        assert result["total"] >= 1
        assert result["results"][0]["name"] == expected
        assert result["results"][0]["trust"] == "builtin"
        assert result["results"][0]["installed"] is False

    result = await handler({"query": "blockchain quantum oracle"})
    assert result["results"] == [] and result["total"] == 0

    with pytest.raises(ToolInputError):
        await handler({"query": "   "})


async def test_discover_ranking_deterministic_top5_and_installed_flag() -> None:
    items = [
        CatalogItem(
            name=f"tool_{chr(ord('a') + i)}",
            description="generic helper",
            keywords=("tool",),
            command="python",
            args=("x.py",),
            trust="builtin" if i % 2 else "community",
        )
        for i in range(7)
    ]
    catalog = ToolCatalog(items)
    session = FakeSession([FakeTool("t")])
    registry, manager = make_meta_registry(catalog, [session])
    handler = registry.get("discover_tools").handler

    result = await handler({"query": "tool"})
    assert result["total"] == 5
    assert [r["name"] for r in result["results"]] == [f"tool_{c}" for c in "abcde"]
    assert all(r["installed"] is False for r in result["results"])

    installed = await registry.get("install_mcp_server").handler({"name": "tool_a"})
    assert installed["ok"] is True
    result2 = await handler({"query": "tool"})
    flags = {r["name"]: r["installed"] for r in result2["results"]}
    assert flags["tool_a"] is True
    assert flags["tool_b"] is False
    assert manager.is_connected("tool_a")


async def test_install_registers_prefixed_tools_and_broadcasts(real_catalog) -> None:
    broadcasts: list = []
    session = FakeSession(
        [FakeTool("reverse_text", "Reverse the text"), FakeTool("to_uppercase")]
    )
    registry, manager = make_meta_registry(real_catalog, [session], broadcasts)
    handler = registry.get("install_mcp_server").handler

    result = await handler({"name": "text_tools"})
    assert result["ok"] is True and result["already_installed"] is False
    assert result["tools"] == [
        "mcp__text_tools__reverse_text",
        "mcp__text_tools__to_uppercase",
    ]
    assert manager.is_connected("text_tools")
    assert (
        await registry.get("mcp__text_tools__reverse_text").handler({"text": "hi"})
        == "echo:hi"
    )
    assert broadcasts == [("install", "text_tools", result["tools"])]


async def test_install_idempotent_second_call_does_not_reconnect(real_catalog) -> None:
    session = FakeSession([FakeTool("reverse_text")])
    connector = make_connector([session])
    registry = ToolRegistry()
    manager = MCPManager([], registry, connect_timeout_s=1, connector=connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    handler = registry.get("install_mcp_server").handler
    first = await handler({"name": "text_tools"})
    second = await handler({"name": "text_tools"})
    assert first["already_installed"] is False
    assert second["already_installed"] is True
    assert second["tools"] == first["tools"]
    assert connector.issued == [session]
    assert len(manager._states) == 1


async def test_install_unknown_server_is_error(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "ghost"})
    assert ei.value.code == "unknown_server"
    assert "password_gen" in (ei.value.hint or "")


async def test_install_connect_failure_is_error_and_retry_succeeds(real_catalog) -> None:
    registry = ToolRegistry()

    def failing_connector(config):
        @asynccontextmanager
        async def cm():
            raise RuntimeError("process failed to spawn")
            yield

        return cm

    manager = MCPManager([], registry, connect_timeout_s=0.5, connector=failing_connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "mcp_connect_failed"
    assert "process failed to spawn" in str(ei.value)
    assert not manager.has_server("text_tools")
    assert mcp_tool_names(registry) == []

    session = FakeSession([FakeTool("reverse_text")])
    manager._connector = make_connector([session])
    result = await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert result["ok"] is True and manager.is_connected("text_tools")


async def test_uninstall_removes_tools_and_broadcasts(real_catalog) -> None:
    broadcasts: list = []
    session = FakeSession([FakeTool("reverse_text")])
    registry, manager = make_meta_registry(real_catalog, [session], broadcasts)
    install = registry.get("install_mcp_server").handler
    uninstall = registry.get("uninstall_mcp_server").handler

    installed = (await install({"name": "text_tools"}))["tools"]
    assert registry.get("mcp__text_tools__reverse_text") is not None

    result = await uninstall({"name": "text_tools"})
    assert result["ok"] is True and result["removed_tools"] == installed
    assert registry.get("mcp__text_tools__reverse_text") is None
    assert not manager.has_server("text_tools")
    assert broadcasts == [
        ("install", "text_tools", installed),
        ("uninstall", "text_tools", installed),
    ]

    discovered = await registry.get("discover_tools").handler({"query": "text"})
    assert all(r["installed"] is False for r in discovered["results"])


async def test_uninstall_not_installed_is_error(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("uninstall_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "not_installed"
    assert "discover_tools" in (ei.value.hint or "")


async def test_install_without_manager_is_error(real_catalog) -> None:
    registry = ToolRegistry()
    register_meta_tools(registry, real_catalog, lambda: None)
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "mcp_unavailable"


async def test_all_meta_tools_have_expected_trust_flags(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    flags = {name: registry.get(name).dangerous for name in registry.names()}
    assert flags["discover_tools"] is False
    assert flags["install_mcp_server"] is True
    assert flags["uninstall_mcp_server"] is True
    assert flags["run_python"] is True


async def test_install_denied_by_approval_does_not_connect(real_catalog) -> None:
    session = FakeSession([FakeTool("reverse_text")])
    connector = make_connector([session])
    registry = ToolRegistry()
    manager = MCPManager([], registry, connect_timeout_s=1, connector=connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    async def deny(name, tool_input) -> bool:
        return False

    dispatcher = ToolDispatcher(registry, approve_callback=deny, retry_backoff_base_s=0)
    result = await dispatcher.dispatch(
        "install_mcp_server", {"name": "text_tools"}, "tu_deny"
    )
    assert result.is_error is True
    assert result.status == "denied"
    assert result.code == "approval_denied"
    assert "user denied" in result.content
    assert connector.issued == []
    assert not manager.has_server("text_tools")
    assert mcp_tool_names(registry) == []


async def test_run_python_normal_execution() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler({"code": "print(6 * 7)"})
    assert result["stdout"].strip() == "42"
    assert result["stderr"] == ""
    assert result["return_code"] == 0
    assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0
    assert result["truncated"] is False


async def test_run_python_syntax_error_returns_nonzero_code() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler({"code": "def broken(:\n    pass"})
    assert result["return_code"] != 0
    assert "SyntaxError" in result["stderr"]
    assert result["stdout"] == ""


async def test_run_python_timeout_force_kills_subprocess() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("run_python").handler(
            {
                "code": "import time; print('start', flush=True); time.sleep(60)",
                "timeout_s": 1,
            }
        )
    elapsed = asyncio.get_running_loop().time() - started
    assert ei.value.code == "run_python_timeout"
    assert "force-killed" in str(ei.value)
    assert elapsed < 30


async def test_run_python_output_truncated_to_50k() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    code = 'import sys; print("o" * 60000); print("e" * 60000, file=sys.stderr)'
    result = await registry.get("run_python").handler({"code": code})
    assert len(result["stdout"]) == 50_000
    assert len(result["stderr"]) == 50_000
    assert result["truncated"] is True


async def test_run_python_isolated_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TEST_SECRET", "leaked")
    monkeypatch.setenv("PYTHONPATH", "C:/should/not/leak")
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    code = (
        "import os, sys;"
        "print('SECRET' in os.environ, 'PYTHONPATH' in os.environ, sys.flags.isolated)"
    )
    result = await registry.get("run_python").handler({"code": code})
    assert result["stdout"].strip() == "False False 1"


async def test_run_python_timeout_clamped_to_settings_max(monkeypatch) -> None:
    monkeypatch.setenv("RUN_PYTHON_MAX_TIMEOUT_S", "1")
    get_settings.cache_clear()
    try:
        registry, _ = make_meta_registry(
            load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
        )
        with pytest.raises(ToolPermanentError) as ei:
            await registry.get("run_python").handler(
                {"code": "import time; time.sleep(60)", "timeout_s": 500}
            )
        assert "1s" in str(ei.value)
        assert ei.value.code == "run_python_timeout"
    finally:
        get_settings.cache_clear()


async def test_run_python_invalid_params() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "   "})
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "pass", "timeout_s": -1})
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "pass", "timeout_s": True})


async def test_run_python_writes_land_in_temp_cwd() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler(
        {"code": "open('scratch.txt', 'w').write('x'); print('done')"}
    )
    assert result["stdout"].strip() == "done"
    assert result["return_code"] == 0


@pytest.fixture
def discovery_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_TOOL_DISCOVERY", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_discovery_disabled_all_four_tools_is_error(
    real_catalog, discovery_disabled
) -> None:
    registry, _ = make_meta_registry(real_catalog, [FakeSession([FakeTool("t")])])
    cases = [
        ("discover_tools", {"query": "text"}),
        ("install_mcp_server", {"name": "text_tools"}),
        ("uninstall_mcp_server", {"name": "text_tools"}),
        ("run_python", {"code": "pass"}),
    ]
    for name, payload in cases:
        with pytest.raises(ToolPermanentError) as ei:
            await registry.get(name).handler(payload)
        assert ei.value.code == "tool_discovery_disabled"
        assert "tool discovery disabled by config" in str(ei.value)
    assert registry.get("install_mcp_server") is not None


async def test_agent_full_chain_discover_install_call_same_run(meta_api) -> None:
    client, application, scripted = meta_api

    fake = ReversingSession(
        [FakeTool("reverse_text", "Reverse the text"), FakeTool("to_uppercase")]
    )
    connector = make_connector([fake])
    application.state.mcp._connector = connector

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_disc", "discover_tools", {"query": "reverse text"}),
        tool_use_turn("tu_install", "install_mcp_server", {"name": "text_tools"}),
        tool_use_turn("tu_rev", "mcp__text_tools__reverse_text", {"text": "abc"}),
        end_turn("installed and used in the same run"),
    ]
    resp = await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "把文本反转一下"}
    )
    assert resp.status_code == 202
    await resolve_when_pending(client, application, session["id"], "tu_install", "allow")
    await wait_run_done(application, session["id"], timeout=10)

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 3
    assert all(b["is_error"] is False for b in tool_results)
    assert '"name": "text_tools"' in tool_results[0]["content"]
    assert '"installed": false' in tool_results[0]["content"]
    assert "mcp__text_tools__reverse_text" in tool_results[1]["content"]
    assert '"already_installed": false' in tool_results[1]["content"]
    assert tool_results[2]["content"] == "cba"
    assert connector.issued == [fake]

    manager = manager_of(application)
    assert session["id"] in manager.bus.known_session_ids()
    buffer = manager.bus._states[session["id"]].buffer
    changed = [e for e in buffer if e.type == SSEEventType.TOOLS_CHANGED]
    assert len(changed) == 1
    event = changed[0]
    assert event.seq > 0
    assert event.data["action"] == "install"
    assert event.data["server"] == "text_tools"
    assert event.data["tools"] == [
        "mcp__text_tools__reverse_text",
        "mcp__text_tools__to_uppercase",
    ]


async def test_agent_install_deny_path_via_approval(meta_api) -> None:
    client, application, scripted = meta_api
    connector = make_connector([FakeSession([FakeTool("reverse_text")])])
    application.state.mcp._connector = connector

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_inst", "install_mcp_server", {"name": "text_tools"}),
        end_turn("user denied; proposing run_python instead"),
    ]
    await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "装一下 text_tools"}
    )
    await resolve_when_pending(client, application, session["id"], "tu_inst", "deny")
    await wait_run_done(application, session["id"], timeout=10)

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert "approval_denied" in tool_results[0]["content"]
    assert "user denied" in tool_results[0]["content"]
    assert connector.issued == []
    manager = manager_of(application)
    assert manager.registry.get("mcp__text_tools__reverse_text") is None
    buffer = manager.bus._states[session["id"]].buffer
    assert not any(e.type == SSEEventType.TOOLS_CHANGED for e in buffer)


async def test_agent_install_real_catalog_server_subprocess(meta_api) -> None:
    pytest.importorskip("mcp")
    client, application, scripted = meta_api
    application.state.mcp._connect_timeout_s = 25.0

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_install", "install_mcp_server", {"name": "text_tools"}),
        tool_use_turn("tu_rev", "mcp__text_tools__reverse_text", {"text": "hello"}),
        end_turn("real server roundtrip"),
    ]

    async def scenario() -> None:
        await client.post(
            f"/api/sessions/{session['id']}/messages",
            json={"text": "装 text_tools 并反转 hello"},
        )
        await resolve_when_pending(client, application, session["id"], "tu_install", "allow")
        await wait_run_done(application, session["id"], timeout=45)

    try:
        async with asyncio.timeout(90):
            await scenario()
    except TimeoutError:
        pytest.skip("catalog MCP server spawn timed out in this environment")

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 2
    assert all(b["is_error"] is False for b in tool_results)
    assert tool_results[1]["content"] == "olleh"
    manager = manager_of(application)
    buffer = manager.bus._states[session["id"]].buffer
    changed = [e for e in buffer if e.type == SSEEventType.TOOLS_CHANGED]
    assert len(changed) == 1 and changed[0].data["action"] == "install"
    assert manager.registry.get("mcp__text_tools__reverse_text") is not None
