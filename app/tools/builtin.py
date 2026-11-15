"""本地工具：read_file / search_workspace / calculate + 工作区管理 ×3
+ write_file / delete_file / run_command（权限控制配套）。

审批门控由 dispatcher 的权限矩阵按 access 类别判定（read 直行；write 类在
read 权限下审批；run_command/run_python 等 execute 类在 write/full 免批）。
read_file / search_workspace / write_file / delete_file 被限制在**当前工作区
目录集合**内，防路径穿越——目录集合经 roots_provider 实时查询（
provider 模式，增删目录即刻生效，无需重注册）。calculate 用 ast 白名单安全
解析，绝不 eval() 任意代码。run_command 复用 run_python 的沙箱纪律（环境变量
白名单、超时钳制、输出截断、Windows 不弹窗）。

工作区管理工具（list_workspace_dirs / add_workspace_dir / remove_workspace_dir）
由 agent 自主调用（按任务需求由模型决策）；受 settings 开关 agent_manage_dirs
控制——关闭时执行返回 is_error("workspace management disabled by config")。
仅在注册时传入 workspace_service（WorkspaceRegistry）时才会注册——没有服务
的场合（部分单测）不向模型暴露无法执行的工具。
"""

from __future__ import annotations

import ast
import asyncio
import operator
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.workspace_registry import (
    WorkspaceError,
    WorkspaceRegistry,
    workspace_dir_payload,
)
from app.tools.errors import ToolInputError, ToolPermanentError
from app.tools.registry import ToolRegistry, ToolSpec

# ---- read_file / search_workspace 的公共防护参数 ----
MAX_READ_BYTES = 200_000
MAX_SEARCH_FILE_BYTES = 1_000_000  # 搜索跳过大文件
MAX_SEARCH_FILES = 2000  # 扫描文件数上限
MAX_SEARCH_RESULTS = 200 # 默认命中数上限

_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".idea", ".vscode", "dist", "build", ".pytest_cache", ".mypy_cache",
    "data", ".DS_Store",
}

_BIN_OP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OP = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_EXPR_LEN = 1000
MAX_RESULT_MAGNITUDE = 10 ** 400  # 防 10**100000 之类的结果爆炸

# roots_provider：工具执行时实时取当前工作区目录集合（provider 模式）
RootsProvider = Callable[[], list[Path]]

MANAGE_DISABLED_MESSAGE = "workspace management disabled by config"


def _coerce_roots_provider(workspace_root: str | Path | RootsProvider) -> RootsProvider:
    """str/Path → 常量 provider；可调用对象原样使用（向后兼容单目录注册）。"""
    if callable(workspace_root):
        return workspace_root
    base = Path(workspace_root)
    return lambda: [base]


def _manage_enabled() -> bool:
    """settings.agent_manage_dirs 开关（调用时读取——测试可动态切换）。"""
    try:
        return bool(get_settings().agent_manage_dirs)
    except Exception:  # pragma: no cover - settings 不可用时保守放行只读工具
        return True


# ----------------------------------------------------------------------
# read_file / search_workspace：多目录解析 + 防穿越
# ----------------------------------------------------------------------


def _resolve_in_roots(roots_provider: RootsProvider, raw_path: str) -> tuple[Path, Path]:
    """把用户路径限制在**任一**当前工作区目录内（防路径穿越）。

    绝对路径、含 .. 的相对路径统一 resolve 后逐一与当前目录集合做包含
    判断，命中第一个包含它的目录；全部不命中即 ToolPermanentError
    （path_outside_workspace，带目录集合线索，永久失败附建议）。
    返回 (目标绝对路径, 命中的根目录)。
    """
    roots = [Path(r).resolve() for r in (roots_provider() or [])]
    candidate = Path(raw_path)
    first_containing: tuple[Path, Path] | None = None
    for root in roots:
        target = candidate if candidate.is_absolute() else root / candidate
        try:
            resolved = target.resolve()
        except OSError:  # pragma: no cover - 非法路径字符等，换下一根目录
            continue
        if resolved == root or resolved.is_relative_to(root):
            # 相对名在多根下可能同时几何包含于多个根——**优先返回文件真实
            # 存在的根**，避免第一个根"遮蔽"后加目录中的同名文件（
            # 真实模式实测缺陷：treasure.txt 被 root0 遮蔽导致 not_found）
            if first_containing is None:
                first_containing = (resolved, root)
            if resolved.exists():
                return resolved, root
    if first_containing is not None:
        # 无一存在 → 回落第一个几何包含的根（报错 hint 沿用该根的兄弟文件线索）
        return first_containing
    hint = (
        "All file access is confined to workspace roots: "
        + ", ".join(str(r) for r in roots)
        if roots
        else "No workspace directories are configured; add one via add_workspace_dir"
    )
    raise ToolPermanentError(
        f"Path escapes workspace roots: {raw_path}",
        code="path_outside_workspace",
        hint=hint,
    )


def _build_read_file(roots_provider: RootsProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        raw_path = tool_input.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolInputError("read_file requires non-empty string field 'path'")
        max_bytes = tool_input.get("max_bytes", MAX_READ_BYTES)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ToolInputError("'max_bytes' must be a positive integer")

        target, root = _resolve_in_roots(roots_provider, raw_path)
        if not target.exists():
            parent = target.parent
            siblings = sorted(p.name for p in parent.iterdir())[:10] if parent.exists() else []
            raise ToolPermanentError(
                f"File not found: {raw_path}",
                code="not_found",
                hint=f"Directory contains: {siblings}" if siblings else "Parent directory is empty or missing",
            )
        if not target.is_file():
            raise ToolPermanentError(
                f"Not a regular file: {raw_path}",
                code="not_a_file",
                hint="read_file only reads regular files; use search_workspace to explore",
            )

        data = target.read_bytes()[:max_bytes]
        truncated = target.stat().st_size > max_bytes
        text = data.decode("utf-8", errors="replace")
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:  # target == root（不会发生，前面已挡非文件）
            rel = Path(raw_path).as_posix()
        return {
            "path": rel,
            "content": text,
            "bytes": len(data),
            "truncated": truncated,
        }

    return ToolSpec(
        name="read_file",
        description="Read a text file inside any current workspace directory. "
        "Input: {path, max_bytes?}.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path—absolute inside any workspace directory, or relative to the first workspace directory"},
                "max_bytes": {"type": "integer", "description": "Max bytes to read (default 200000)"},
            },
            "required": ["path"],
        },
        handler=handler,
        timeout_s=15.0,
        parallel_safe=True,
        dangerous=False,
        access="read",
    )


# ----------------------------------------------------------------------
# search_workspace
# ----------------------------------------------------------------------


def _iter_workspace_files(root: Path, seen: set[Path], budget: list[int]):
    """遍历一个 workspace 目录（跳过隐藏/依赖目录与点文件），产出文件路径。

    seen：已产出文件的 resolved 集合（嵌套目录间去重）；budget：跨目录
    共享的扫描配额（[剩余数]，原地扣减），总数封顶 MAX_SEARCH_FILES。
    """
    for path in sorted(root.rglob("*")):
        if budget[0] <= 0:
            return
        if path.is_dir():
            continue
        if any(
            part in _SKIP_DIRS or part.startswith(".")
            for part in path.relative_to(root).parts
        ):
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        budget[0] -= 1
        yield path, root


def _build_search_workspace(roots_provider: RootsProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        pattern = tool_input.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ToolInputError("search_workspace requires non-empty string field 'pattern'")
        glob = tool_input.get("glob")
        if glob is not None and not isinstance(glob, str):
            raise ToolInputError("'glob' must be a string like '*.py'")
        case_insensitive = tool_input.get("case_insensitive", True)
        if not isinstance(case_insensitive, bool):
            raise ToolInputError("'case_insensitive' must be a boolean")
        max_results = tool_input.get("max_results", MAX_SEARCH_RESULTS)
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
            raise ToolInputError("'max_results' must be a positive integer")

        needle = pattern.lower() if case_insensitive else pattern
        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        seen: set[Path] = set()
        budget = [MAX_SEARCH_FILES]
        for root in roots_provider() or []:
            for path, home in _iter_workspace_files(root.resolve(), seen, budget):
                if glob is not None and not path.match(glob):
                    continue
                try:
                    if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                scanned += 1
                hay = text.lower() if case_insensitive else text
                if needle not in hay:
                    continue
                rel = path.relative_to(home).as_posix()
                for line_no, line in enumerate(text.splitlines(), start=1):
                    probe = line.lower() if case_insensitive else line
                    if needle in probe:
                        matches.append(
                            {"file": rel, "line": line_no, "text": line.strip()[:200]}
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
                if truncated:
                    break
            if truncated:
                break
        return {
            "matches": matches,
            "total": len(matches),
            "files_scanned": scanned,
            "truncated": truncated,
        }

    return ToolSpec(
        name="search_workspace",
        description="Keyword-search text files inside all current workspace directories "
        "(grep-like). Input: {pattern, glob?, case_insensitive?, max_results?}.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Keyword to search for"},
                "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'"},
                "case_insensitive": {"type": "boolean", "description": "Default true"},
                "max_results": {"type": "integer", "description": "Max matches returned (default 200)"},
            },
            "required": ["pattern"],
        },
        handler=handler,
        timeout_s=30.0,
        parallel_safe=True,
        dangerous=False,
        access="read",
    )


# ----------------------------------------------------------------------
# calculate
# ----------------------------------------------------------------------


def _safe_eval(node: ast.AST) -> int | float:
    """ast 白名单求值：仅数字常量与 + - * / // % ** 和一元 +/-。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolInputError(
                f"Only numeric literals are allowed, got: {node.value!r}",
                hint="Allowed: numbers, + - * / // % **, parentheses",
            )
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OP:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        try:
            value = _BIN_OP[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolPermanentError("Division by zero", code="zero_division") from exc
        except (OverflowError, ValueError) as exc:
            raise ToolPermanentError(
                f"Arithmetic overflow in expression", code="arithmetic_overflow"
            ) from exc
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ToolPermanentError("Result is not finite", code="arithmetic_overflow")
        if isinstance(value, int) and abs(value) > MAX_RESULT_MAGNITUDE:
            raise ToolPermanentError("Result magnitude exceeds limit", code="arithmetic_overflow")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OP:
        return _UNARY_OP[type(node.op)](_safe_eval(node.operand))
    raise ToolInputError(
        f"Unsupported syntax: {type(node).__name__}",
        hint="Allowed: numbers, + - * / // % **, parentheses",
    )


def _build_calculate() -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        expression = tool_input.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolInputError("calculate requires non-empty string field 'expression'")
        if len(expression) > MAX_EXPR_LEN:
            raise ToolInputError(f"Expression too long (max {MAX_EXPR_LEN} chars)")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolInputError(
                f"Invalid arithmetic expression: {exc.msg}",
                hint="Allowed: numbers, + - * / // % **, parentheses",
            ) from exc
        result = _safe_eval(tree)
        return {"expression": expression, "result": result}

    return ToolSpec(
        name="calculate",
        description="Evaluate an arithmetic expression safely. "
        "Input: {expression} with numbers and + - * / // % ** parentheses.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '(2 + 3) * 7 / 5'"},
            },
            "required": ["expression"],
        },
        handler=handler,
        timeout_s=5.0,
        parallel_safe=True,
        dangerous=False,
        access="read",
    )

def _guard_manage_enabled() -> None:
    """agent_manage_dirs=false 时三个管理工具统一拒绝（is_error 链路）。"""
    if not _manage_enabled():
        raise ToolPermanentError(
            MANAGE_DISABLED_MESSAGE,
            code="workspace_management_disabled",
            hint="Set agent_manage_dirs=true in config.yaml to enable",
        )


def _raise_workspace_error(exc: WorkspaceError) -> None:
    """WorkspaceRegistry 的领域错误 → 工具永久失败（code 原样透传）。"""
    raise ToolPermanentError(str(exc), code=exc.code, hint=exc.hint) from exc


def _build_list_workspace_dirs(service: WorkspaceRegistry) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_manage_enabled()
        dirs = [workspace_dir_payload(r) for r in service.snapshot()]
        return {"dirs": dirs, "count": len(dirs)}

    return ToolSpec(
        name="list_workspace_dirs",
        description="List the current workspace directories that read_file / "
        "search_workspace can access. Input: {}.",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
        timeout_s=5.0,
        parallel_safe=True,
        dangerous=False,
        access="read",
    )


def _build_add_workspace_dir(service: WorkspaceRegistry) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_manage_enabled()
        raw = tool_input.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ToolInputError("add_workspace_dir requires non-empty string field 'path'")
        try:
            record = await service.add(raw, source="agent")
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return {
            "ok": True,
            "dir": workspace_dir_payload(record),
            "message": f"Directory added to workspace ({record.source}): {record.path}",
        }

    return ToolSpec(
        name="add_workspace_dir",
        description="Add an existing absolute directory path to the workspace so "
        "read_file / search_workspace can access it. Fails on duplicate or "
        "non-directory paths. Input: {path}.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path, e.g. D:\\data"},
            },
            "required": ["path"],
        },
        handler=handler,
        timeout_s=10.0,
        parallel_safe=False,
        dangerous=False,
        access="write",  # 变更工作区目录集合，read 权限下需审批
    )


def _build_remove_workspace_dir(service: WorkspaceRegistry) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        _guard_manage_enabled()
        raw = tool_input.get("path_or_id")
        if not isinstance(raw, str) or not raw.strip():
            raise ToolInputError(
                "remove_workspace_dir requires non-empty string field 'path_or_id'"
            )
        try:
            record = await service.remove_by_ref(raw)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return {
            "ok": True,
            "removed": workspace_dir_payload(record),
            "count": len(service.snapshot()),
            "message": f"Directory removed from workspace: {record.path}",
        }

    return ToolSpec(
        name="remove_workspace_dir",
        description="Remove a workspace directory by its id or path (the last "
        "remaining directory cannot be removed). Input: {path_or_id}.",
        input_schema={
            "type": "object",
            "properties": {
                "path_or_id": {"type": "string", "description": "Directory id or path from list_workspace_dirs"},
            },
            "required": ["path_or_id"],
        },
        handler=handler,
        timeout_s=10.0,
        parallel_safe=False,  # 变更共享目录集合，串行化
        dangerous=False,
        access="write",
    )


# ----------------------------------------------------------------------
# write_file / delete_file（文件写删工具，access=write）
# ----------------------------------------------------------------------

MAX_WRITE_BYTES = 2_000_000  # 单次写入字节上限（防撑爆磁盘/上下文）
_RUN_CMD_DEFAULT_TIMEOUT_S = 30.0
_RUN_CMD_MAX_OUTPUT_CHARS = 50_000
_RUN_CMD_GRACE_S = 5.0

# 命令子进程环境变量白名单（与 meta_tools.run_python 同纪律：父进程的
# 自定义变量——密钥/令牌等——绝不透传给任意命令执行）
_RUN_ENV_WHITELIST = (
    "PATH", "PATHEXT", "COMSPEC", "SystemRoot", "windir",
    "TEMP", "TMP", "TMPDIR", "HOME",
)


def _build_write_file(roots_provider: RootsProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        raw_path = tool_input.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolInputError("write_file requires non-empty string field 'path'")
        content = tool_input.get("content")
        if not isinstance(content, str):
            raise ToolInputError("write_file requires string field 'content'")
        overwrite = tool_input.get("overwrite", True)
        if not isinstance(overwrite, bool):
            raise ToolInputError("'overwrite' must be a boolean")

        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolInputError(
                f"Content too large: {len(data)} bytes (max {MAX_WRITE_BYTES})",
                hint="Split the write into multiple files or smaller chunks",
            )

        target, root = _resolve_in_roots(roots_provider, raw_path)
        if target.is_dir():
            raise ToolPermanentError(
                f"Target is a directory: {raw_path}",
                code="not_a_file",
                hint="write_file writes regular files only",
            )
        if target.exists() and not overwrite:
            raise ToolPermanentError(
                f"File already exists: {raw_path}",
                code="file_exists",
                hint="Pass overwrite=true to replace, or choose another name",
            )
        created = not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - resolve 后必在 root 内
            rel = Path(raw_path).as_posix()
        return {
            "path": rel,
            "bytes": len(data),
            "created": created,
            "message": f"{'Created' if created else 'Overwrote'} {rel} ({len(data)} bytes)",
        }

    return ToolSpec(
        name="write_file",
        description="Create or overwrite a text file inside any current workspace "
        "directory (parent directories are created automatically). "
        "Input: {path, content, overwrite?}.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path—absolute inside any workspace directory, or relative to the first workspace directory"},
                "content": {"type": "string", "description": "Full text content to write (UTF-8, max 2MB)"},
                "overwrite": {"type": "boolean", "description": "Overwrite if exists (default true)"},
            },
            "required": ["path", "content"],
        },
        handler=handler,
        timeout_s=10.0,
        parallel_safe=True,
        dangerous=True,
        access="write",
    )


def _build_delete_file(roots_provider: RootsProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        raw_path = tool_input.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolInputError("delete_file requires non-empty string field 'path'")

        target, root = _resolve_in_roots(roots_provider, raw_path)
        if target in {Path(r).resolve() for r in (roots_provider() or [])}:
            raise ToolPermanentError(
                f"Refusing to delete a workspace root: {raw_path}",
                code="workspace_root",
                hint="Remove it via remove_workspace_dir instead",
            )
        if not target.exists():
            raise ToolPermanentError(
                f"File not found: {raw_path}",
                code="not_found",
                hint="Use search_workspace to locate the exact path",
            )
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            if any(target.iterdir()):
                raise ToolPermanentError(
                    f"Directory not empty: {raw_path}",
                    code="dir_not_empty",
                    hint="delete_file only removes empty directories or files; "
                    "use run_command for recursive deletion",
                )
            target.rmdir()
        else:  # pragma: no cover - 特殊文件（socket 等）
            raise ToolPermanentError(
                f"Not a regular file or directory: {raw_path}",
                code="not_a_file",
            )
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover
            rel = Path(raw_path).as_posix()
        return {"path": rel, "deleted": True, "message": f"Deleted {rel}"}

    return ToolSpec(
        name="delete_file",
        description="Delete a file or an EMPTY directory inside any current workspace "
        "directory (workspace roots themselves are protected). "
        "Input: {path}.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path—absolute inside any workspace directory, or relative to the first workspace directory"},
            },
            "required": ["path"],
        },
        handler=handler,
        timeout_s=10.0,
        parallel_safe=True,
        dangerous=True,
        access="write",
    )


# ----------------------------------------------------------------------
# run_command（bash/命令执行，access=execute）
# ----------------------------------------------------------------------


def _resolve_shell_command() -> list[str]:
    """跨平台 shell 前缀：优先 bash（Git Bash / WSL），Windows 回落 cmd。"""
    bash = shutil.which("bash")
    if bash:
        return [bash, "-c"]
    if os.name == "nt":
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c"]
    return ["/bin/sh", "-c"]


def _no_window_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _build_run_command(roots_provider: RootsProvider) -> ToolSpec:
    async def handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolInputError("run_command requires non-empty string field 'command'")
        timeout_s = tool_input.get("timeout_s", _RUN_CMD_DEFAULT_TIMEOUT_S)
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ToolInputError("'timeout_s' must be a positive number")
        try:
            effective = min(float(timeout_s), float(get_settings().run_python_max_timeout_s))
        except Exception:  # pragma: no cover - settings 不可用时用入参
            effective = float(timeout_s)
        effective = max(1.0, effective)

        roots = [Path(r).resolve() for r in (roots_provider() or [])]
        if not roots:
            raise ToolPermanentError(
                "No workspace directories configured",
                code="no_workspace",
                hint="Add one via add_workspace_dir first",
            )
        cwd = roots[0]
        raw_cwd = tool_input.get("cwd")
        if isinstance(raw_cwd, str) and raw_cwd.strip():
            cwd, _ = _resolve_in_roots(roots_provider, raw_cwd)
            if not cwd.is_dir():
                raise ToolPermanentError(
                    f"cwd is not a directory: {raw_cwd}",
                    code="not_a_directory",
                )

        env = {k: v for k, v in os.environ.items() if k in _RUN_ENV_WHITELIST}
        argv = [*_resolve_shell_command(), command]
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                creationflags=_no_window_flags(),
            )
        except OSError as exc:
            raise ToolPermanentError(
                f"Failed to spawn shell: {exc}",
                code="spawn_failed",
            ) from exc
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=effective)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:  # pragma: no cover - 已自行退出
                pass
            try:
                await asyncio.wait_for(proc.communicate(), timeout=_RUN_CMD_GRACE_S)
            except Exception:  # pragma: no cover - 管道回收失败不掩盖超时
                pass
            raise ToolPermanentError(
                f"run_command timed out after {effective:g}s; subprocess force-killed",
                code="timeout",
                hint="Use a larger timeout_s or split the command into steps",
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        truncated = len(stdout) > _RUN_CMD_MAX_OUTPUT_CHARS or len(stderr) > _RUN_CMD_MAX_OUTPUT_CHARS
        return {
            "stdout": stdout[:_RUN_CMD_MAX_OUTPUT_CHARS],
            "stderr": stderr[:_RUN_CMD_MAX_OUTPUT_CHARS],
            "return_code": proc.returncode,
            "duration_ms": duration_ms,
            "truncated": truncated,
            "cwd": cwd.as_posix(),
        }

    return ToolSpec(
        name="run_command",
        description="Run a shell command (bash if available, cmd on Windows fallback) "
        "inside a workspace directory. Sandboxed: whitelisted env, hard timeout, "
        "truncated output. Input: {command, timeout_s?, cwd?}.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line to execute"},
                "timeout_s": {"type": "number", "description": "Execution timeout in seconds (default 30, hard-capped by config)"},
                "cwd": {"type": "string", "description": "Working directory—absolute inside any workspace directory, or relative to the first one (default: first workspace directory)"},
            },
            "required": ["command"],
        },
        handler=handler,
        # dispatcher 的强杀要晚于沙箱自身的超时钳制（max + 余量）
        timeout_s=float(get_settings().run_python_max_timeout_s) + 30.0,
        parallel_safe=True,  # 独立子进程，互不干扰
        dangerous=True,
        access="execute",
    )


# ----------------------------------------------------------------------


def register_default_tools(
    registry: ToolRegistry,
    workspace_root: str | Path | RootsProvider,
    workspace_service: WorkspaceRegistry | None = None,
) -> list[ToolSpec]:
    """注册默认安全工具；返回按名排序的注册结果。

    workspace_root：目录集合来源——str/Path（兼容旧单目录注册）或
    roots_provider 回调（provider 模式，执行时实时查询）。
    workspace_service：WorkspaceRegistry 实例；提供时额外注册工作区管理
    三工具（list / add / remove），agent 自主增删即刻生效。
    """
    roots_provider = _coerce_roots_provider(workspace_root)
    specs = [
        _build_read_file(roots_provider),
        _build_search_workspace(roots_provider),
        _build_calculate(),
        _build_write_file(roots_provider),
        _build_delete_file(roots_provider),
        _build_run_command(roots_provider),
    ]
    if workspace_service is not None:
        specs.extend(
            [
                _build_list_workspace_dirs(workspace_service),
                _build_add_workspace_dir(workspace_service),
                _build_remove_workspace_dir(workspace_service),
            ]
        )
    for spec in specs:
        registry.register(spec)
    return sorted(specs, key=lambda s: s.name)
