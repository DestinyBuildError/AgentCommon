
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

RUN_PYTHON_DEFAULT_TIMEOUT_S = 30.0
RUN_PYTHON_MAX_OUTPUT_CHARS = 50_000
_RUN_PYTHON_GRACE_S = 5.0

_RUN_ENV_WHITELIST = (
    "PATH", "PATHEXT", "COMSPEC", "SystemRoot", "windir",
    "TEMP", "TMP", "TMPDIR", "HOME",
)

DISCOVERY_DISABLED_MESSAGE = "tool discovery disabled by config"
MCP_UNAVAILABLE_MESSAGE = "MCP manager is not available in this process"

ToolsChangeCallback = Callable[[str, str, list[str]], Awaitable[None]]
MCPProvider = Callable[[], Any]


def _discovery_enabled() -> bool:
    try:
        return bool(get_settings().enable_tool_discovery)
    except Exception:
        return True


def _guard_discovery_enabled() -> None:
    if not _discovery_enabled():
        raise ToolPermanentError(
            DISCOVERY_DISABLED_MESSAGE,
            code="tool_discovery_disabled",
            hint="Set enable_tool_discovery=true in config.yaml to enable",
        )


def _require_manager(mcp_provider: MCPProvider) -> Any:
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
    except Exception:
        return set()


def _installed_flag_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"query_matched": len(results), "results": results}


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
        access="read",
    )


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
        timeout_s=60.0,
        parallel_safe=False,
        dangerous=True,
        access="admin",
    )


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
        parallel_safe=False,
        dangerous=True,
        access="admin",
    )


def _max_run_timeout_s() -> float:
    try:
        value = float(get_settings().run_python_max_timeout_s)
    except Exception:
        return 120.0
    return value if value > 0 else 120.0


def _sandbox_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _RUN_ENV_WHITELIST}


def _creation_flags() -> int:
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
        timeout_s=_max_run_timeout_s() + 30.0,
        parallel_safe=True,
        dangerous=True,
        access="execute",
    )


async def _force_kill(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        return
    except Exception:
        pass
    try:
        await asyncio.wait_for(proc.communicate(), timeout=_RUN_PYTHON_GRACE_S)
    except Exception:
        pass


async def _notify(
    on_change: ToolsChangeCallback | None,
    action: str,
    server: str,
    tools: list[str],
) -> None:
    if on_change is None:
        return
    try:
        await on_change(action, server, list(tools))
    except asyncio.CancelledError:
        raise
    except Exception:
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
    specs = [
        _build_discover_tools(catalog, mcp_provider),
        _build_install_mcp_server(catalog, mcp_provider, on_change),
        _build_uninstall_mcp_server(mcp_provider, on_change),
        _build_run_python(),
    ]
    for spec in specs:
        registry.register(spec)
    return specs
