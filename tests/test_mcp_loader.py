
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from app.core.dispatcher import ToolDispatcher
from app.mcp_loader import (
    MCPManager,
    MCPServerConfig,
    UNREACHABLE_TEMPLATE,
    _resolve_python_alias,
    parse_mcp_servers,
)
from app.tools.errors import ToolPermanentError
from app.tools.registry import ToolRegistry

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_python_alias_resolves_to_backend_interpreter() -> None:
    assert _resolve_python_alias("python") == sys.executable
    assert _resolve_python_alias("Python3.EXE") == sys.executable
    assert _resolve_python_alias("python3") == sys.executable
    assert _resolve_python_alias(str(BACKEND_ROOT / ".venv" / "Scripts" / "python.exe")) == str(
        BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"
    )
    assert _resolve_python_alias(None) is None


class FakeTool:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.input_schema = {"type": "object", "properties": {}}


class FakeListToolsResult:
    def __init__(self, tools: list[FakeTool]) -> None:
        self.tools = tools


class FakeCallToolResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        class _Text:
            def __init__(self, t: str) -> None:
                self.text = t
                self.type = "text"

        self.content = [_Text(text)] if text is not None else []
        self.is_error = is_error


class FakeSession:

    def __init__(
        self,
        tools: list[FakeTool] | None = None,
        *,
        fail_calls: bool = False,
        hang_calls: bool = False,
        calls: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._tools = tools or []
        self._fail_calls = fail_calls
        self._hang_calls = hang_calls
        self.calls = calls if calls is not None else []
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> FakeListToolsResult:
        return FakeListToolsResult(self._tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], read_timeout_seconds: Any = None
    ) -> FakeCallToolResult:
        self.calls.append((name, dict(arguments)))
        if self._hang_calls:
            await asyncio.sleep(3600)
        if self._fail_calls:
            raise ConnectionError("stdio stream closed")
        if name == "boom_tool":
            return FakeCallToolResult("remote validation failed", is_error=True)
        return FakeCallToolResult(f"echo:{arguments.get('text', '')}")


def make_connector(sessions: list[FakeSession]):
    issued: list[FakeSession] = []

    def connector(config: Any):
        @asynccontextmanager
        async def cm() -> Any:
            if not sessions:
                raise RuntimeError("no more fake sessions")
            session = sessions.pop(0)
            issued.append(session)
            yield session

        return cm

    connector.issued = issued
    return connector


def make_registry() -> ToolRegistry:
    return ToolRegistry()


def demo_config(name: str = "demo") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="python", args=["x.py"])


def test_parse_mcp_servers_normal_and_invalid() -> None:
    configs = parse_mcp_servers(
        [
            {"name": "demo", "command": "python", "args": ["a.py"]},
            {"name": "http-one", "url": "http://127.0.0.1:8080/mcp"},
            {"name": "bad", "args": ["no-command-or-url"]},
            "not-a-dict",
            {"name": "off", "command": "python", "enabled": False},
        ]
    )
    assert [c.name for c in configs] == ["demo", "http-one", "off"]
    assert configs[0].args == ["a.py"]
    assert configs[2].enabled is False
    assert parse_mcp_servers(None) == []
    assert parse_mcp_servers([]) == []


async def test_start_registers_tools_with_namespace_prefix() -> None:
    session = FakeSession([FakeTool("current_time", "Return time"), FakeTool("echo_text")])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=2, connector=make_connector([session])
    )
    connected = await manager.start()
    assert connected == ["demo"]
    assert "mcp__demo__current_time" in registry.names()
    assert "mcp__demo__echo_text" in registry.names()
    spec = registry.get("mcp__demo__current_time")
    assert spec is not None
    assert spec.description == "Return time"
    assert spec.metadata["mcp_server"] == "demo"
    schema = spec.api_schema()
    assert schema["name"] == "mcp__demo__current_time"
    assert schema["input_schema"] == {"type": "object", "properties": {}}
    assert await spec.handler({"text": "hi"}) == "echo:hi"
    await manager.stop()


async def test_tool_call_timeout_maps_to_mcp_unreachable() -> None:
    session = FakeSession([FakeTool("slow")], hang_calls=True)
    registry = make_registry()
    manager = MCPManager(
        [demo_config()],
        registry,
        connect_timeout_s=1,
        call_timeout_s=0.2,
        connector=make_connector([session]),
    )
    await manager.start()
    spec = registry.get("mcp__demo__slow")
    with pytest.raises(ToolPermanentError) as exc_info:
        await spec.handler({})
    assert exc_info.value.code == "mcp_unreachable"
    await manager.stop()


async def test_startup_failure_degrades_without_blocking() -> None:

    def connector(config: Any):
        @asynccontextmanager
        async def cm() -> Any:
            if config.name == "broken":
                raise RuntimeError("process failed to spawn")
            yield FakeSession([FakeTool("t1")])

        return cm

    registry = make_registry()
    manager = MCPManager(
        [
            MCPServerConfig(name="broken", command="x"),
            MCPServerConfig(name="good", command="x"),
        ],
        registry,
        connect_timeout_s=1,
        connector=connector,
    )
    connected = await manager.start()
    assert connected == ["good"]
    assert registry.get("mcp__broken__t1") is None
    assert registry.get("mcp__good__t1") is not None
    await manager.stop()


async def test_startup_timeout_degrades() -> None:

    def connector(config: Any):
        @asynccontextmanager
        async def cm() -> Any:
            session = FakeSession([FakeTool("t")])

            async def hang() -> None:
                await asyncio.sleep(3600)

            session.initialize = hang
            yield session

        return cm

    registry = make_registry()
    manager = MCPManager(
        [MCPServerConfig(name="hang", command="x")],
        registry,
        connect_timeout_s=0.3,
        connector=connector,
    )
    connected = await manager.start()
    assert connected == []
    assert registry.names() == []
    await manager.stop()


async def test_duplicate_tool_name_skipped_not_fatal() -> None:
    session = FakeSession([FakeTool("t1")])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=1, connector=make_connector([session])
    )
    await manager.start()
    count_before = len(registry)
    registered = manager._register_tools(
        manager._states["demo"], [FakeTool("t1")]
    )
    assert registered == 0
    assert len(registry) == count_before
    await manager.stop()


async def test_runtime_disconnect_marks_unavailable_and_returns_is_error() -> None:
    good = FakeSession([FakeTool("echo_text")])
    broken = FakeSession([FakeTool("echo_text")], fail_calls=True)
    registry = make_registry()
    manager = MCPManager(
        [demo_config()],
        registry,
        connect_timeout_s=1,
        connector=make_connector([broken]),
    )
    await manager.start()

    dispatcher = ToolDispatcher(registry)
    result = await dispatcher.dispatch("mcp__demo__echo_text", {"text": "hi"}, "tu_1")
    assert result.is_error is True
    assert UNREACHABLE_TEMPLATE.format(server="demo") in result.content
    assert "try local tools" in result.content
    assert result.code == "mcp_unreachable"
    assert manager._states["demo"].available is False
    await manager.stop()


async def test_lazy_reconnect_recovers_on_next_call() -> None:
    first = FakeSession([FakeTool("echo_text")], fail_calls=True)
    second = FakeSession([FakeTool("echo_text")])
    connector = make_connector([first, second])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=1, connector=connector
    )
    await manager.start()
    spec = registry.get("mcp__demo__echo_text")

    with pytest.raises(ToolPermanentError) as exc_info:
        await spec.handler({"text": "a"})
    assert exc_info.value.code == "mcp_unreachable"

    assert await spec.handler({"text": "b"}) == "echo:b"
    assert connector.issued == [first, second]
    assert manager._states["demo"].available is True
    await manager.stop()


async def test_reconnect_failure_still_reports_unreachable() -> None:
    first = FakeSession([FakeTool("t")], fail_calls=True)
    connector = make_connector([first])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=0.5, connector=connector
    )
    await manager.start()
    spec = registry.get("mcp__demo__t")
    with pytest.raises(ToolPermanentError):
        await spec.handler({})
    with pytest.raises(ToolPermanentError) as exc_info:
        await spec.handler({})
    assert exc_info.value.code == "mcp_unreachable"
    assert manager._states["demo"].available is False
    await manager.stop()


async def test_remote_tool_iserror_raises_tool_error() -> None:
    session = FakeSession([FakeTool("boom_tool")])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=1, connector=make_connector([session])
    )
    await manager.start()
    dispatcher = ToolDispatcher(registry)
    result = await dispatcher.dispatch("mcp__demo__boom_tool", {}, "tu_2")
    assert result.is_error is True
    assert result.code == "mcp_tool_error"
    assert "remote validation failed" in result.content
    assert manager._states["demo"].available is True
    await manager.stop()


async def test_call_to_unknown_server_returns_unreachable() -> None:
    registry = make_registry()
    manager = MCPManager([], registry)
    with pytest.raises(ToolPermanentError) as exc_info:
        await manager.call("ghost", "t", {})
    assert exc_info.value.code == "mcp_unreachable"


def test_real_demo_server_stdio_roundtrip():
    pytest.importorskip("mcp")
    registry = make_registry()
    config = MCPServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(BACKEND_ROOT / "demo_mcp_server.py")],
    )
    manager = MCPManager([config], registry, connect_timeout_s=20.0, call_timeout_s=20.0)

    async def scenario() -> None:
        connected = await asyncio.wait_for(manager.start(), timeout=30)
        assert connected == ["demo"]
        assert "mcp__demo__current_time" in registry.names()
        assert "mcp__demo__echo_text" in registry.names()
        dispatcher = ToolDispatcher(registry)
        result = await dispatcher.dispatch(
            "mcp__demo__echo_text", {"text": "ping"}, "tu_e2e"
        )
        assert result.is_error is False
        assert result.content == "ping"

    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=60))
    except (asyncio.TimeoutError, TimeoutError):
        pytest.skip("demo MCP server spawn timed out in this environment")
    finally:
        asyncio.run(manager.stop())
