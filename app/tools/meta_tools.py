"""元工具：自扩展 Agent——发现/安装/卸载 MCP server + Python 沙箱兜底。

闭环：模型发现没有所需工具时，先 discover_tools 在精选目录
（tool_catalog.yaml）里搜索可安装的 MCP server；install_mcp_server 运行时
连接该 server 并把工具注册进 ToolRegistry——agent_loop 每次迭代都重新调
registry.api_schemas()，所以**同一 run 的下一迭代立即可调用新工具**；
uninstall_mcp_server 断开并移除。run_python 是万能兜底：没有工具就直接
写 Python 代码干（沙箱子进程）。

信任模型（安全设计 / 已批准的设计结论）：
- discover_tools dangerous=False——纯目录检索，无副作用，只读自动放行
- install / uninstall / run_python dangerous=True——一律过 H5 审批门，
  防提示注入诱导安装恶意工具或执行任意代码
- 四个工具全部受 settings.enable_tool_discovery 开关控制：关闭 →
  执行时统一 is_error("tool discovery disabled by config")（与
  agent_manage_dirs 的 workspace 管理开关同一模式）

注册风格照抄 workspace 三工具：_build_* 工厂 + ToolSpec + provider 模式——
mcp_provider 回调在**执行时**才取 MCPManager（规避 lifespan 里
SessionManager 先于 MCPManager 构建的顺序问题），on_change 广播回调由
app 层注入（复用 workspace_changed 的 bus.emit + known_session_ids 模式）。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import get_settings
from app.tools.catalog import SEARCH_TOP_K, CatalogItem, ToolCatalog
from app.tools.errors import ToolInputError, ToolPermanentError
from app.tools.registry import ToolRegistry, ToolSpec

# ---- run_python 沙箱参数 ----
RUN_PYTHON_DEFAULT_TIMEOUT_S = 30.0   # 模型未传 timeout_s 时的默认值
RUN_PYTHON_MAX_OUTPUT_CHARS = 50_000  # stdout / stderr 各自截断上限
_RUN_PYTHON_GRACE_S = 5.0             # 强杀后回收管道的宽限

# -I（隔离模式：忽略 PYTHON* 环境变量、不用 user site）之外的额外隔离：
# 子进程环境变量走白名单——父进程的自定义变量（密钥/令牌等）绝不透传，
# 仅保留子进程正常启动所需的系统级变量。
_RUN_ENV_WHITELIST = (
    "PATH", "PATHEXT", "COMSPEC", "SystemRoot", "windir",
    "TEMP", "TMP", "TMPDIR", "HOME",
)

DISCOVERY_DISABLED_MESSAGE = "tool discovery disabled by config"
MCP_UNAVAILABLE_MESSAGE = "MCP manager is not available in this process"

# 广播回调：async (action: "install"|"uninstall", server: str, tools: list[str]) -> None
ToolsChangeCallback = Callable[[str, str, list[str]], Awaitable[None]]
# MCP 管理器提供者：() -> MCPManager | None（provider 模式，执行时取）
MCPProvider = Callable[[], Any]


def _discovery_enabled() -> bool:
    """settings.enable_tool_discovery 开关（调用时读取——测试可动态切换）。"""
    try:
        return bool(get_settings().enable_tool_discovery)
    except Exception:  # pragma: no cover - settings 不可用时保守放行
        return True


def _guard_discovery_enabled() -> None:
    """开关关闭 → 四个元工具统一拒绝（is_error 链路）。"""
    if not _discovery_enabled():
        raise ToolPermanentError(
            DISCOVERY_DISABLED_MESSAGE,
            code="tool_discovery_disabled",
            hint="Set enable_tool_discovery=true in config.yaml to enable",
        )


def _require_manager(mcp_provider: MCPProvider) -> Any:
    """provider 取 MCPManager；取不到（理论不可达）→ 永久失败。"""
    manager = mcp_provider()
    if manager is None:
        raise ToolPermanentError(
            MCP_UNAVAILABLE_MESSAGE,
            code="mcp_unavailable",
            hint="The MCP layer was not initialized at startup; check backend logs",
        )
    return manager


def _installed_names(manager: Any) -> set[str]:
    try:
        return set(manager.installed_server_names())
    except Exception:  # pragma: no cover - 防御：广播层失败不影响只读发现
        return set()


def _installed_flag_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"query_matched": len(results), "results": results}


# ----------------------------------------------------------------------
# discover_tools
# ----------------------------------------------------------------------


def _build_discover_tools(catalog: ToolCatalog, mcp_provider: MCPProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_discovery_enabled()
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("discover_tools requires non-empty string field 'query'")
        manager = mcp_provider()
        installed = _installed_names(manager) if manager is not None else set()
        results = catalog.search(query, installed, top_k=SEARCH_TOP_K)
        return {
            "results": results,
            "total": len(results),
            "catalog_size": len(catalog.all()),
            "message": (
                f"{len(results)} installable server(s) matched. "
                "Call install_mcp_server(name) to install one; its tools become "
                "immediately available in the next iteration."
                if results
                else "No installable server matched the query. "
                "Use run_python as a fallback."
            ),
        }

    return ToolSpec(
        name="discover_tools",
        description="Search the curated tool catalog for installable MCP servers "
        "by keyword (matches name/description/keywords, case-insensitive). "
        "Use this when you lack a tool for the task. Input: {query}.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords, e.g. 'password', '温度换算'"},
            },
            "required": ["query"],
        },
        handler=handler,
        timeout_s=5.0,
        parallel_safe=True,
        dangerous=False,
        access="read",  # 纯目录检索，无副作用
    )


# ----------------------------------------------------------------------
# install_mcp_server
# ----------------------------------------------------------------------


def _build_install_mcp_server(
    catalog: ToolCatalog,
    mcp_provider: MCPProvider,
    on_change: ToolsChangeCallback | None,
) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_discovery_enabled()
        name = tool_input.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolInputError("install_mcp_server requires non-empty string field 'name'")
        item: CatalogItem | None = catalog.get(name.strip())
        if item is None:
            raise ToolPermanentError(
                f"Unknown catalog server: {name}",
                code="unknown_server",
                hint=f"Catalog contains: {catalog.names()}",
            )
        manager = _require_manager(mcp_provider)
        # 幂等：已装 → 提示即可（不算错误，不重复连接）
        if manager.is_connected(item.name):
            tools = manager.registered_tool_names(item.name)
            return {
                "ok": True,
                "already_installed": True,
                "server": item.name,
                "tools": tools,
                "message": f"MCP server '{item.name}' is already installed "
                f"({len(tools)} tool(s)); its tools are ready to call.",
            }
        result = await manager.connect_server(item.to_server_config())
        if not result.ok:
            raise ToolPermanentError(
                f"Failed to install MCP server '{item.name}': {result.error}",
                code="mcp_connect_failed",
                hint="The server process may have crashed at startup; "
                "check its transport in tool_catalog.yaml and backend logs",
            )
        await _notify(on_change, "install", item.name, result.tools)
        return {
            "ok": True,
            "already_installed": False,
            "server": item.name,
            "tools": result.tools,
            "message": f"MCP server '{item.name}' installed; {len(result.tools)} "
            f"tool(s) registered ({', '.join(result.tools) or 'none'}). "
            "They are callable right now.",
        }

    return ToolSpec(
        name="install_mcp_server",
        description="Install an MCP server found via discover_tools: connect to it "
        "at runtime and register its tools (named mcp__<server>__<tool>) so they "
        "are callable in the very next step. Input: {name}.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Server name from discover_tools results"},
            },
            "required": ["name"],
        },
        handler=handler,
        timeout_s=60.0,  # 覆盖 manager 的 10s 连接超时 + tools/list 余量
        parallel_safe=False,  # 变更共享工具注册表，串行化
        dangerous=True,
        access="admin",  # 元操作：运行时安装 MCP server，仅 full 免批
    )


# ----------------------------------------------------------------------
# uninstall_mcp_server
# ----------------------------------------------------------------------


def _build_uninstall_mcp_server(
    mcp_provider: MCPProvider,
    on_change: ToolsChangeCallback | None,
) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_discovery_enabled()
        name = tool_input.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolInputError("uninstall_mcp_server requires non-empty string field 'name'")
        manager = _require_manager(mcp_provider)
        key = name.strip()
        tools = manager.registered_tool_names(key)
        removed = await manager.disconnect_server(key)
        if not removed:
            raise ToolPermanentError(
                f"MCP server '{key}' is not installed",
                code="not_installed",
                hint="Use discover_tools to list installable servers and "
                "install_mcp_server to install one",
            )
        await _notify(on_change, "uninstall", key, tools)
        return {
            "ok": True,
            "server": key,
            "removed_tools": tools,
            "message": f"MCP server '{key}' uninstalled; {len(tools)} tool(s) removed.",
        }

    return ToolSpec(
        name="uninstall_mcp_server",
        description="Uninstall a previously installed MCP server: disconnect from it "
        "and remove its tools from the registry. Input: {name}.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Installed server name"},
            },
            "required": ["name"],
        },
        handler=handler,
        timeout_s=15.0,
        parallel_safe=False,  # 变更共享工具注册表，串行化
        dangerous=True,
        access="admin",  # 元操作：运行时卸载 MCP server，仅 full 免批
    )


# ----------------------------------------------------------------------
# run_python
# ----------------------------------------------------------------------


def _max_run_timeout_s() -> float:
    try:
        value = float(get_settings().run_python_max_timeout_s)
    except Exception:  # pragma: no cover - settings 不可用时用默认上限
        return 120.0
    return value if value > 0 else 120.0


def _sandbox_env() -> dict[str, str]:
    """白名单环境变量 → 子进程读不到父进程的自定义变量（配合 -I 双保险）。"""
    return {k: v for k, v in os.environ.items() if k in _RUN_ENV_WHITELIST}


def _creation_flags() -> int:
    """Windows 后台进程不弹控制台窗口；POSIX 无此概念返回 0。"""
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _build_run_python() -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_discovery_enabled()
        code = tool_input.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ToolInputError("run_python requires non-empty string field 'code'")
        timeout_s = tool_input.get("timeout_s", RUN_PYTHON_DEFAULT_TIMEOUT_S)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ToolInputError("'timeout_s' must be a positive number")
        max_timeout = _max_run_timeout_s()
        effective = min(float(timeout_s), max_timeout)

        started = time.monotonic()
        # 隔离临时目录为 CWD：代码默认只能看见自己产生的文件，不落在工作区
        with tempfile.TemporaryDirectory(prefix="run_python_", ignore_cleanup_errors=True) as workdir:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", code,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_sandbox_env(),
                creationflags=_creation_flags(),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=effective
                )
            except asyncio.TimeoutError:
                await _force_kill(proc)
                raise ToolPermanentError(
                    f"run_python timed out after {effective:g}s; subprocess force-killed",
                    code="run_python_timeout",
                    hint="Reduce the workload or pass a larger timeout_s "
                    f"(hard cap {max_timeout:g}s)",
                )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        truncated = len(stdout) > RUN_PYTHON_MAX_OUTPUT_CHARS or len(stderr) > RUN_PYTHON_MAX_OUTPUT_CHARS
        return {
            "stdout": stdout[:RUN_PYTHON_MAX_OUTPUT_CHARS],
            "stderr": stderr[:RUN_PYTHON_MAX_OUTPUT_CHARS],
            "return_code": proc.returncode,
            "duration_ms": duration_ms,
            "truncated": truncated,
        }

    return ToolSpec(
        name="run_python",
        description="Run Python code in an isolated sandbox subprocess as a "
        "general-purpose fallback when no dedicated tool exists. Input: "
        "{code, timeout_s?} (timeout default 30s, hard-capped).",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute; print() to produce output"},
                "timeout_s": {"type": "number", "description": "Execution timeout in seconds (default 30)"},
            },
            "required": ["code"],
        },
        handler=handler,
        # dispatcher 的强杀要晚于沙箱自身的超时钳制（max + 余量）
        timeout_s=_max_run_timeout_s() + 30.0,
        parallel_safe=True,  # 独立子进程，互不干扰
        dangerous=True,
        access="execute",  # 任意代码执行，write/full 免批、read 需审批
    )


async def _force_kill(proc: asyncio.subprocess.Process) -> None:
    """超时强杀并回收管道（杀不掉不掩盖超时错误本身）。"""
    try:
        proc.kill()
    except ProcessLookupError:  # pragma: no cover - 已自行退出
        return
    except Exception:  # pragma: no cover - 平台差异兜底
        pass
    try:
        # 回收被强杀进程的管道输出，防僵尸/管道阻塞
        await asyncio.wait_for(proc.communicate(), timeout=_RUN_PYTHON_GRACE_S)
    except Exception:  # pragma: no cover
        pass


# ----------------------------------------------------------------------


async def _notify(
    on_change: ToolsChangeCallback | None,
    action: str,
    server: str,
    tools: list[str],
) -> None:
    """广播 tools_changed；广播层失败不影响安装/卸载本身（best-effort）。"""
    if on_change is None:
        return
    try:
        await on_change(action, server, list(tools))
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - 广播失败只记日志
        import logging

        logging.getLogger(__name__).warning(
            "tools_changed broadcast failed for %s(%s)", server, action
        )


def register_meta_tools(
    registry: ToolRegistry,
    catalog: ToolCatalog,
    mcp_provider: MCPProvider,
    on_change: ToolsChangeCallback | None = None,
) -> list[ToolSpec]:
    """注册四个元工具（discover / install / uninstall / run_python）。

    mcp_provider：执行时取 MCPManager 的回调（provider 模式——lifespan 里
    SessionManager 先于 MCPManager 构建，注册时还拿不到实例）。
    on_change：install/uninstall 成功后的广播回调（app 层接 event_bus）。
    """
    specs = [
        _build_discover_tools(catalog, mcp_provider),
        _build_install_mcp_server(catalog, mcp_provider, on_change),
        _build_uninstall_mcp_server(mcp_provider, on_change),
        _build_run_python(),
    ]
    for spec in specs:
        registry.register(spec)
    return specs
