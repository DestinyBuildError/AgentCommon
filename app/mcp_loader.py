"""MCP 接入层（生命周期失败隔离）。

职责：
- 从 config（mcp_servers 列表：name/command/args 或 url）逐个连接 server
  （stdio 走官方 mcp SDK 的 stdio_client；url 走 streamablehttp_client）
- 拉取各 server 的 tools/list，转成 ToolSpec 注册进全局 ToolRegistry，
  命名空间前缀 `mcp__<server>__<tool>`（不同 server 工具重名不冲突；
  模型视角只有一套工具，MCP 只是 dispatcher 的另一种路由目标）
- MCP 工具 handler = 调用该 server 的 tools/call（默认超时 60s）

失败隔离（策略表）：
- 启动时 server 起不来 / 超时（默认 10s）→ **降级不阻塞**：跳过 + warning，
  其余 server 继续
- 运行中 server 崩溃 / 无响应 → 该 server 标记 unavailable；后续调用返回
  is_error("MCP server X unreachable, try local tools")；下次调用前惰性
  重连一次（重连失败仍报同一错误，不重试风暴）

连接模型：每个 server 一个长驻连接任务（owns the async context managers，
anyio cancel scope 要求 enter/exit 同 task），工具调用从其他任务经
ClientSession 发请求。连接任务死亡（进程崩溃/流断开）→ done-callback 标记
unavailable，惰性重连时重建任务。测试可注入 connector（fake session），
绝不依赖真实第三方 server。

（自扩展 Agent）新增：connect_server / disconnect_server 支持
**运行时**动态安装/卸载单个 server（install_mcp_server / uninstall_mcp_server
元工具的底层）——连接任务挂当前运行循环，注册/注销走既有
mcp__<server>__<tool> 前缀；失败语义与启动降级一致（返回错误不抛出）。
"""

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

CONNECT_TIMEOUT_S = 10.0   # 启动/（惰性）重连超时
CALL_TIMEOUT_S = 60.0      # tools/call 默认超时
STOP_GRACE_S = 5.0         # 优雅关闭等待上限，超时强杀

UNREACHABLE_TEMPLATE = "MCP server {server} unreachable, try local tools"

# 裸 python 别名 → 解析为当前后端解释器（venv，必有 mcp 依赖）。
# PATH 上的 "python" 在 Windows 常是另一个裸解释器（无 mcp），demo server
# 会起不来；显式写绝对路径时原样使用，不走本别名。
_PYTHON_ALIASES = {"python", "python3", "python.exe", "python3.exe"}


def _resolve_python_alias(command: str | None) -> str | None:
    if command is not None and command.strip().lower() in _PYTHON_ALIASES:
        return sys.executable
    return command

# 连接器：config → async CM（yield session-like 对象，需实现
# initialize()/list_tools()/call_tool()）。生产默认用 mcp SDK；测试注入 fake。
Connector = Callable[[Any], Any]


@dataclass(frozen=True)
class MCPServerConfig:
    """单个 MCP server 的连接配置（来自 config.yaml 的 mcp_servers 项）。"""

    name: str
    command: str | None = None       # stdio 模式：可执行文件
    args: list[str] = field(default_factory=list)
    url: str | None = None           # http 模式：streamable http 端点
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None           # stdio 子进程工作目录
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
    """config.yaml 的 mcp_servers 原始列表 → 配置对象；非法项跳过 + warning。"""
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
    """connect_server 的归一化结果（运行时安装）。

    失败不抛出——ok=False + error 文本返回给调用方（install_mcp_server
    再转成 is_error 的 tool_result），与启动降级语义一致。
    """

    ok: bool
    server: str
    tools: list[str] = field(default_factory=list)  # 注册的工具名（含 mcp__ 前缀）
    already_connected: bool = False                 # 幂等安装：已在连接中
    error: str | None = None


@dataclass
class _ServerState:
    """单个 server 的运行时状态。"""

    config: MCPServerConfig
    session: Any | None = None
    task: asyncio.Task | None = None   # 长驻连接任务（owns contexts）
    stop_event: asyncio.Event | None = None
    available: bool = False


def _stdio_connector(config: MCPServerConfig):
    """生产连接器：stdio（command/args）或 streamable http（url）。

    返回 async CM，yield 初始化就绪的 ClientSession；CM 退出时按序关闭
    session 与传输（同一任务内，满足 anyio cancel scope 约束）。
    """
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
    """MCP server 生命周期与工具桥接。

    connector 可注入（测试）：Callable(config) -> async CM yielding
    session-like 对象（initialize/list_tools/call_tool）。
    """

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

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    async def start(self) -> list[str]:
        """连接全部 server 并注册工具（启动失败降级不阻塞）。

        返回成功连接的 server 名列表。重复调用幂等（先停再起）。
        """
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
                # 降级不阻塞：跳过该 server，其余继续（第一行）
                logger.warning(
                    "MCP server %s failed to start; skipped: %r", config.name, exc
                )
                state.available = False
                await self._teardown(state)
        return connected

    async def stop(self) -> None:
        """停掉全部连接任务（进程收尾 / manager 复位用）。"""
        for state in self._states.values():
            await self._teardown(state)

    # ------------------------------------------------------------------
    # 运行时动态安装 / 卸载（自扩展 Agent）
    # ------------------------------------------------------------------

    async def connect_server(self, config: MCPServerConfig) -> MCPConnectResult:
        """运行时连接单个 server 并把其工具注册进 ToolRegistry。

        长驻连接任务用 asyncio.create_task 挂**当前运行循环**（与启动路径
        共用 _connect）——agent_loop 每次迭代都重调 registry.api_schemas()，
        所以本方法返回后**同一 run 的下一迭代立即可调用新工具**。

        失败语义与启动降级一致：返回 ok=False + error 文本，不抛出
        （CancelledError 原样上抛除外）。已连接 → 幂等成功（already_connected）；
        残留死状态（启动失败/断连遗留）→ 先清理再重连；失败时清理现场
        （状态出表，下次 install 可干净重试）。
        """
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
        """断开单个 server 并从 ToolRegistry 注销其全部工具。

        server 不存在 → False（"未装"由调用方提示）；存在则连状态带工具
        一并移除——连接任务优雅停止，工具即刻从 api_schemas() 消失。
        """
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
        """是否有该 server 的状态条目（含死状态）。"""
        return name in self._states

    def is_connected(self, name: str) -> bool:
        """该 server 当前是否处于可用连接（幂等安装 / installed 标记的依据）。"""
        state = self._states.get(name)
        return state is not None and self._is_alive(state)

    def registered_tool_names(self, server_name: str) -> list[str]:
        """某 server 已注册进 registry 的工具名（mcp__<server>__ 前缀，排序确定）。"""
        prefix = f"mcp__{server_name}__"
        return sorted(n for n in self._registry.names() if n.startswith(prefix))

    def installed_server_names(self) -> list[str]:
        """当前处于可用连接的 server 名（discover_tools 的 installed 标记用）。"""
        return sorted(n for n, s in self._states.items() if self._is_alive(s))

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    async def _connect(self, state: _ServerState) -> Any:
        """起长驻连接任务并等 initialize 完成（限时），返回可用 session。"""
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._run_connection(state, ready), name=f"mcp-conn:{state.config.name}"
        )
        task.add_done_callback(lambda t, s=state: self._on_connection_done(s, t))
        state.task = task
        try:
            session = await asyncio.wait_for(ready, timeout=self._connect_timeout_s)
        except BaseException:
            # 启动超时/失败：连接任务自行退出（ready 已带异常）或强杀
            if not task.done():
                task.cancel()
            raise
        state.session = session
        return session

    async def _run_connection(self, state: _ServerState, ready: asyncio.Future) -> None:
        """连接任务：own 全部 async contexts，initialize 后挂起等 stop。"""
        config = state.config
        state.stop_event = asyncio.Event()
        try:
            async with self._connector(config)() as session:
                await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout_s)
                ready.set_result(session)
                await state.stop_event.wait()  # 挂起直到 stop/teardown
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
        """连接任务死亡（崩溃/断流）→ 标记 unavailable（第二行）。"""
        state.available = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and state.session is not None:
            logger.warning("MCP server %s disconnected: %r", state.config.name, exc)

    async def _teardown(self, state: _ServerState) -> None:
        """优雅停止连接任务；挂起不退则强杀。

        注意：await 一个已被取消的子任务会向本协程抛 CancelledError——
        用 task.done() 区分「子任务取消完成（正常收尾，吞掉）」与
        「本协程被外部取消（原样上抛）」。
        """
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
                # 外部取消：确保子任务也停，然后原样上抛
                task.cancel()
                raise
            # 子任务已取消完成（_connect 超时路径的 task.cancel()）：正常收尾
        except (asyncio.TimeoutError, Exception):
            # 优雅期超时 / 收尾异常 → 强杀，确保任务消亡
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    def _is_alive(self, state: _ServerState) -> bool:
        return state.available and state.task is not None and not state.task.done()

    # ------------------------------------------------------------------
    # 工具调用桥（handler 侧）
    # ------------------------------------------------------------------

    async def call(self, server_name: str, remote_tool: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具，返回文本结果；失败抛 ToolError（dispatcher 归一化）。

        - 连接不在 → 惰性重连一次；仍失败 → mcp_unreachable
        - 调用异常/超时 → 标记 unavailable + mcp_unreachable
        - server 返回 isError → mcp_tool_error（内容透传给模型自纠）
        """
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
                timeout=self._call_timeout_s + 5.0,  # 外层兜底，防 read_timeout 失效
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 运行中断连 / 无响应 → 标记 unavailable（第二行）
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
        """惰性重连一次（每次调用前至多一次，不做重试风暴）。"""
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

    # ------------------------------------------------------------------
    # 工具注册
    # ------------------------------------------------------------------

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
        """MCP tool 定义 → ToolSpec（命名空间前缀 mcp__<server>__<tool>）。"""

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
            access="execute",  # MCP 工具副作用不可静态判定，按命令执行类归档
            metadata={"mcp_server": server_name, "mcp_tool": remote_name},
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        """CallToolResult.content（MCP content blocks）→ 纯文本。"""
        parts: list[str] = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                # 非文本块（image/audio 等）降级为占位描述
                btype = getattr(block, "type", None)
                if btype:
                    parts.append(f"[non-text content block: {btype}]")
        return "\n".join(parts).strip()
