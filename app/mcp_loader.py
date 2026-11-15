
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from app.tools.errors import ToolPermanentError
from app.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10.0
CALL_TIMEOUT_S = 60.0
STOP_GRACE_S = 5.0

UNREACHABLE_TEMPLATE = "MCP server {server} unreachable, try local tools"

_PYTHON_ALIASES = {"python", "python3", "python.exe", "python3.exe"}


def _resolve_python_alias(command: str | None) -> str | None:
    if command is not None and command.strip().lower() in _PYTHON_ALIASES:
        return sys.executable
    return command

Connector = Callable[[Any], Any]


@dataclass(frozen=True)
class MCPServerConfig:

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MCPServerConfig:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("mcp_servers entry missing required field 'name'")
        command = raw.get("command")
        url = raw.get("url")
        if not command and not url:
            raise ValueError(f"mcp_servers[{name}]: needs 'command' (stdio) or 'url' (http)")
        args = raw.get("args") or []
        if not isinstance(args, list):
            raise ValueError(f"mcp_servers[{name}]: 'args' must be a list")
        env = raw.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"mcp_servers[{name}]: 'env' must be a mapping")
        return cls(
            name=name,
            command=str(command) if command else None,
            args=[str(a) for a in args],
            url=str(url) if url else None,
            env={str(k): str(v) for k, v in env.items()},
            cwd=str(raw["cwd"]) if raw.get("cwd") else None,
            enabled=bool(raw.get("enabled", True)),
        )


def parse_mcp_servers(raw_list: Any) -> list[MCPServerConfig]:
    if not raw_list:
        return []
    if not isinstance(raw_list, list):
        logger.warning("mcp_servers must be a list; ignoring")
        return []
    configs: list[MCPServerConfig] = []
    for raw in raw_list:
        try:
            if not isinstance(raw, dict):
                raise ValueError("entry must be a mapping")
            configs.append(MCPServerConfig.from_dict(raw))
        except ValueError as exc:
            logger.warning("invalid mcp_servers entry skipped: %s", exc)
    return configs


@dataclass(frozen=True)
class MCPConnectResult:

    ok: bool
    server: str
    tools: list[str] = field(default_factory=list)
    already_connected: bool = False
    error: str | None = None


@dataclass
class _ServerState:

    config: MCPServerConfig
    session: Any | None = None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None
    available: bool = False


def _stdio_connector(config: MCPServerConfig):
    command = _resolve_python_alias(config.command)

    @asynccontextmanager
    async def _connect() -> AsyncIterator[Any]:
        if config.url:
            from mcp.client.streamable_http import streamablehttp_client
            from mcp import ClientSession as _CS

            async with streamablehttp_client(config.url) as (read, write, _):
                async with _CS(read, write) as session:
                    yield session
        else:
            from mcp import ClientSession as _CS, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=command or "",
                args=list(config.args),
                env=config.env or None,
                cwd=config.cwd,
            )
            async with stdio_client(params) as (read, write):
                async with _CS(read, write) as session:
                    yield session

    return _connect


class MCPManager:

    def __init__(
        self,
        configs: list[MCPServerConfig],
        registry: ToolRegistry,
        *,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
        call_timeout_s: float = CALL_TIMEOUT_S,
        connector: Connector | None = None,
    ) -> None:
        self._configs = [c for c in configs if c.enabled]
        self._registry = registry
        self._connect_timeout_s = connect_timeout_s
        self._call_timeout_s = call_timeout_s
        self._connector = connector or _stdio_connector
        self._states: dict[str, _ServerState] = {}
        self._started = False


    async def start(self) -> list[str]:
        if self._started:
            await self.stop()
        self._started = True
        connected: list[str] = []
        for config in self._configs:
            state = _ServerState(config=config)
            self._states[config.name] = state
            try:
                session = await self._connect(state)
                tools = await asyncio.wait_for(
                    session.list_tools(), timeout=self._connect_timeout_s
                )
                registered = self._register_tools(state, getattr(tools, "tools", []))
                state.available = True
                connected.append(config.name)
                logger.info(
                    "MCP server %s connected; registered %d tool(s)", config.name, registered
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "MCP server %s failed to start; skipped: %r", config.name, exc
                )
                state.available = False
                await self._teardown(state)
        return connected

    async def stop(self) -> None:
        for state in self._states.values():
            await self._teardown(state)


    async def connect_server(self, config: MCPServerConfig) -> MCPConnectResult:
        existing = self._states.get(config.name)
        if existing is not None and self._is_alive(existing):
            return MCPConnectResult(
                ok=True,
                server=config.name,
                tools=self.registered_tool_names(config.name),
                already_connected=True,
            )
        if existing is not None:
            await self._teardown(existing)
            self._states.pop(config.name, None)

        state = _ServerState(config=config)
        self._states[config.name] = state
        try:
            session = await self._connect(state)
            tools = await asyncio.wait_for(
                session.list_tools(), timeout=self._connect_timeout_s
            )
            self._register_tools(state, getattr(tools, "tools", []))
            state.available = True
            names = self.registered_tool_names(config.name)
            logger.info(
                "MCP server %s connected at runtime; registered %d tool(s)",
                config.name,
                len(names),
            )
            return MCPConnectResult(ok=True, server=config.name, tools=names)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "MCP server %s failed to connect at runtime: %r", config.name, exc
            )
            state.available = False
            await self._teardown(state)
            self._states.pop(config.name, None)
            return MCPConnectResult(
                ok=False, server=config.name, error=str(exc) or repr(exc)
            )

    async def disconnect_server(self, name: str) -> bool:
        state = self._states.pop(name, None)
        if state is None:
            return False
        await self._teardown(state)
        removed = 0
        for tool_name in self.registered_tool_names(name):
            if self._registry.unregister(tool_name):
                removed += 1
        if removed:
            logger.info(
                "MCP server %s disconnected at runtime; unregistered %d tool(s)",
                name,
                removed,
            )
        return True

    def has_server(self, name: str) -> bool:
        return name in self._states

    def is_connected(self, name: str) -> bool:
        state = self._states.get(name)
        return state is not None and self._is_alive(state)

    def registered_tool_names(self, server_name: str) -> list[str]:
        prefix = f"mcp__{server_name}__"
        return sorted(n for n in self._registry.names() if n.startswith(prefix))

    def installed_server_names(self) -> list[str]:
        return sorted(n for n, s in self._states.items() if self._is_alive(s))


    async def _connect(self, state: _ServerState) -> Any:
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._run_connection(state, ready), name=f"mcp-conn:{state.config.name}"
        )
        task.add_done_callback(lambda t, s=state: self._on_connection_done(s, t))
        state.task = task
        try:
            session = await asyncio.wait_for(ready, timeout=self._connect_timeout_s)
        except BaseException:
            if not task.done():
                task.cancel()
            raise
        state.session = session
        return session

    async def _run_connection(self, state: _ServerState, ready: asyncio.Future) -> None:
        config = state.config
        state.stop_event = asyncio.Event()
        try:
            async with self._connector(config)() as session:
                await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout_s)
                ready.set_result(session)
                await state.stop_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("MCP server %s connection closed: %r", config.name, exc)
            raise
        finally:
            if not ready.done():
                ready.set_exception(RuntimeError("connection closed before initialize"))

    def _on_connection_done(self, state: _ServerState, task: asyncio.Task) -> None:
        state.available = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and state.session is not None:
            logger.warning("MCP server %s disconnected: %r", state.config.name, exc)

    async def _teardown(self, state: _ServerState) -> None:
        state.available = False
        state.session = None
        if state.stop_event is not None:
            state.stop_event.set()
        task = state.task
        state.task = None
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=STOP_GRACE_S)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                raise
        except (asyncio.TimeoutError, Exception):
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    def _is_alive(self, state: _ServerState) -> bool:
        return state.available and state.task is not None and not state.task.done()


    async def call(self, server_name: str, remote_tool: str, arguments: dict[str, Any]) -> str:
        state = self._states.get(server_name)
        if state is None:
            raise ToolPermanentError(
                UNREACHABLE_TEMPLATE.format(server=server_name),
                code="mcp_unreachable",
                hint="The MCP server was skipped at startup (see backend logs).",
            )
        if not self._is_alive(state):
            await self._reconnect(state)
        session = state.session
        try:
            result = await asyncio.wait_for(
                session.call_tool(
                    remote_tool, dict(arguments), read_timeout_seconds=self._call_timeout_s
                ),
                timeout=self._call_timeout_s + 5.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.available = False
            await self._teardown(state)
            raise ToolPermanentError(
                UNREACHABLE_TEMPLATE.format(server=server_name),
                code="mcp_unreachable",
            ) from exc
        text = self._extract_text(result)
        if getattr(result, "is_error", False):
            raise ToolPermanentError(
                text or f"MCP tool '{remote_tool}' returned an error",
                code="mcp_tool_error",
            )
        return text

    async def _reconnect(self, state: _ServerState) -> None:
        logger.info("MCP server %s unavailable; attempting lazy reconnect", state.config.name)
        await self._teardown(state)
        try:
            session = await self._connect(state)
            await asyncio.wait_for(session.list_tools(), timeout=self._connect_timeout_s)
            state.available = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.available = False
            raise ToolPermanentError(
                UNREACHABLE_TEMPLATE.format(server=state.config.name),
                code="mcp_unreachable",
            ) from exc


    def _register_tools(self, state: _ServerState, tools: list[Any]) -> int:
        registered = 0
        for tool in tools or []:
            remote_name = str(getattr(tool, "name", "") or "")
            if not remote_name:
                continue
            spec = self._build_spec(state.config.name, remote_name, tool)
            if self._registry.get(spec.name) is not None:
                logger.warning(
                    "MCP tool %s collides with existing tool; skipped", spec.name
                )
                continue
            self._registry.register(spec)
            registered += 1
        return registered

    def _build_spec(self, server_name: str, remote_name: str, tool: Any) -> ToolSpec:

        async def handler(tool_input: dict[str, Any]) -> str:
            return await self.call(server_name, remote_name, dict(tool_input))

        schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
        return ToolSpec(
            name=f"mcp__{server_name}__{remote_name}",
            description=str(getattr(tool, "description", None) or remote_name),
            input_schema=dict(schema),
            handler=handler,
            timeout_s=self._call_timeout_s,
            parallel_safe=True,
            dangerous=False,
            access="execute",
            metadata={"mcp_server": server_name, "mcp_tool": remote_name},
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                btype = getattr(block, "type", None)
                if btype:
                    parts.append(f"[non-text content block: {btype}]")
        return "\n".join(parts).strip()
