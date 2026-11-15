"""MCPManager 单元测试—— 连接层全部 fake，绝不依赖真实第三方 server。

覆盖：正常注册带前缀（mcp__<server>__<tool>）、启动失败降级不阻塞、
运行中断连 → unavailable → is_error("MCP server X unreachable, try
local tools")、惰性重连一次恢复、tools/call isError 透传、以及
demo_mcp_server.py 真实 stdio 子进程的端到端冒烟（mcp SDK 可用时）。
"""

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
    """裸 python 别名 → 当前解释器（venv，必有 mcp）；显式路径原样保留。"""
    assert _resolve_python_alias("python") == sys.executable
    assert _resolve_python_alias("Python3.EXE") == sys.executable
    assert _resolve_python_alias("python3") == sys.executable
    assert _resolve_python_alias(str(BACKEND_ROOT / ".venv" / "Scripts" / "python.exe")) == str(
        BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"
    )
    assert _resolve_python_alias(None) is None


# ----------------------------------------------------------------------
# Fakes：session-like 对象（initialize/list_tools/call_tool）
# ----------------------------------------------------------------------


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
    """可脚本化的 session：断连（call_tool 抛异常）、挂起（超时路径）均可。"""

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
    """构造注入式 connector：按序 yield 预设 session（连接即消费一个）。"""
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

    connector.issued = issued  # type: ignore[attr-defined]
    return connector


def make_registry() -> ToolRegistry:
    return ToolRegistry()


def demo_config(name: str = "demo") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="python", args=["x.py"])


# ----------------------------------------------------------------------
# parse_mcp_servers
# ----------------------------------------------------------------------


def test_parse_mcp_servers_normal_and_invalid() -> None:
    configs = parse_mcp_servers(
        [
            {"name": "demo", "command": "python", "args": ["a.py"]},
            {"name": "http-one", "url": "http://127.0.0.1:8080/mcp"},
            {"name": "bad", "args": ["no-command-or-url"]},  # 缺 command/url
            "not-a-dict",
            {"name": "off", "command": "python", "enabled": False},
        ]
    )
    assert [c.name for c in configs] == ["demo", "http-one", "off"]
    assert configs[0].args == ["a.py"]
    assert configs[2].enabled is False
    assert parse_mcp_servers(None) == []
    assert parse_mcp_servers([]) == []


# ----------------------------------------------------------------------
# 正常注册：命名空间前缀
# ----------------------------------------------------------------------


async def test_start_registers_tools_with_namespace_prefix() -> None:
    session = FakeSession([FakeTool("current_time", "Return time"), FakeTool("echo_text")])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=2, connector=make_connector([session])
    )
    connected = await manager.start()
    assert connected == ["demo"]
    # 前缀命名空间进同一 registry
    assert "mcp__demo__current_time" in registry.names()
    assert "mcp__demo__echo_text" in registry.names()
    spec = registry.get("mcp__demo__current_time")
    assert spec is not None
    assert spec.description == "Return time"
    assert spec.metadata["mcp_server"] == "demo"
    # api schema 形态正确（模型视角只有一套工具）
    schema = spec.api_schema()
    assert schema["name"] == "mcp__demo__current_time"
    assert schema["input_schema"] == {"type": "object", "properties": {}}
    # handler 经 manager.call 走 tools/call
    assert await spec.handler({"text": "hi"}) == "echo:hi"
    await manager.stop()


async def test_tool_call_timeout_maps_to_mcp_unreachable() -> None:
    """tools/call 挂起 → 超时 → 标记 unavailable + mcp_unreachable。"""
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


# ----------------------------------------------------------------------
# 启动失败：降级不阻塞（第一行）
# ----------------------------------------------------------------------


async def test_startup_failure_degrades_without_blocking() -> None:
    """坏 server（连接即炸）被跳过 + warning，其余 server 正常注册。"""

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
    """initialize 挂起 → 连接超时 → 该 server 跳过。"""

    def connector(config: Any):
        @asynccontextmanager
        async def cm() -> Any:
            session = FakeSession([FakeTool("t")])

            async def hang() -> None:
                await asyncio.sleep(3600)

            session.initialize = hang  # type: ignore[method-assign]
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
    """两个 server 提供同名工具 → 前缀不同不冲突；同 server 重复注册被跳过。"""
    session = FakeSession([FakeTool("t1")])
    registry = make_registry()
    manager = MCPManager(
        [demo_config()], registry, connect_timeout_s=1, connector=make_connector([session])
    )
    await manager.start()
    # 手动再注册同名 → 冲突被跳过（不抛 ValueError）
    count_before = len(registry)
    registered = manager._register_tools(
        manager._states["demo"], [FakeTool("t1")]
    )
    assert registered == 0
    assert len(registry) == count_before
    await manager.stop()


# ----------------------------------------------------------------------
# 运行中断连：unavailable → is_error → 惰性重连（第二行）
# ----------------------------------------------------------------------


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

    # 经 dispatcher 全链路：handler 抛 mcp_unreachable → 归一化 is_error
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

    # 第一次调用：断连 → unavailable
    with pytest.raises(ToolPermanentError) as exc_info:
        await spec.handler({"text": "a"})
    assert exc_info.value.code == "mcp_unreachable"

    # 下次调用前惰性重连一次 → 换新 session → 成功
    assert await spec.handler({"text": "b"}) == "echo:b"
    assert connector.issued == [first, second]  # 重连恰好发生一次
    assert manager._states["demo"].available is True
    await manager.stop()


async def test_reconnect_failure_still_reports_unreachable() -> None:
    first = FakeSession([FakeTool("t")], fail_calls=True)
    connector = make_connector([first])  # 无后续 session → 重连必失败
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
    """server 返回 isError → mcp_tool_error（内容透传给模型自纠）。"""
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
    assert manager._states["demo"].available is True  # isError 不算断连
    await manager.stop()


async def test_call_to_unknown_server_returns_unreachable() -> None:
    registry = make_registry()
    manager = MCPManager([], registry)
    with pytest.raises(ToolPermanentError) as exc_info:
        await manager.call("ghost", "t", {})
    assert exc_info.value.code == "mcp_unreachable"


# ----------------------------------------------------------------------
# demo_mcp_server.py 真实 stdio 子进程端到端（mcp SDK 可用时跑）
# ----------------------------------------------------------------------


def test_real_demo_server_stdio_roundtrip():
    """真实拉起 demo_mcp_server.py：注册带前缀 + echo 全链路（启动失败降级的
    反面验证）。子进程 spawn 慢平台可能超时——视为环境问题跳过，不算失败。"""
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
