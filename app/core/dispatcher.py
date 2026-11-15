"""工具调度层：失败归一化、超时、内部重试、熔断、并行隔离、审批门。

核心原则：错误不逃逸出循环 —— 任何一层失败都归一化为
ToolResult（is_error + 结构化 JSON 错误文本），由模型决策换路。

失败分类学 → 归一化映射（七类）：
  1 参数不合法 → invalid_params （schema 提示，模型自纠；不计熔断）
  2 未知工具 → unknown_tool （列出可用工具；不计熔断）
  3 可重试失败 → 按 spec.retryable 内部重试（默认 0 次），耗尽 → transient_failure
  4 永久失败 → 立即 error，附可执行建议 hint
  5 超时 → asyncio.wait_for 强杀 → timeout after Ns
  6 审批拒绝 → approval_denied："user denied, do NOT retry, propose alternative"
  7 部分成功 → 工具自行返回如实的完成/未完成状态（工具设计尽量幂等，
                   dispatcher 原样透传，不做二次加工）

熔断器：同工具连续执行失败 ≥ tool_breaker_limit（默认 3）→ 熔断，
后续调用直接返回 breaker_tripped，不再执行、不再审批；成功一次即清零。
审批拒绝 / 参数错误 / 未知工具不算执行失败（不是工具本身的锅）。

并行隔离：dispatch_many 用 asyncio.gather(return_exceptions=True)；
parallel_safe=True 的工具并发执行，False 的经全局锁串行化。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.tools.errors import ToolError, ToolTransientError, tool_error
from app.tools.registry import ACCESS_LEVELS, ToolRegistry, ToolSpec

BREAKER_HINT = "Stop retrying this tool. Try a different approach or ask the user."
DENIED_MESSAGE = "user denied, do NOT retry, propose alternative"
DEFAULT_BREAKER_LIMIT = 3
DEFAULT_TIMEOUT_S = 60.0

# 基础退避（秒）：第 n 次重试前睡 backoff_base * 2**(n-1)；测试可置 0
RETRY_BACKOFF_BASE_S = 0.1

# ---- 权限控制（全局开关由前端经 PUT /api/permissions 切换）----
# 三档权限模式；dispatcher 默认 read（最保守）——直接构造 dispatcher 的
# 旧测试行为不变（只读直行、危险工具审批），生产路径由 SessionManager
# 从持久化设置装载真实模式。
PERMISSION_MODES = ("read", "write", "full")
DEFAULT_PERMISSION_MODE = "read"

# 权限矩阵：mode → access → 动作（allow 直接执行 / approval 走审批门 /
# deny 拒绝执行）。矩阵是权限系统的唯一事实来源：
# read : 只读直行；文件写删 / 命令执行 / 元操作一律审批
# write : 读 + 文件写删 + 命令执行直行；仅元操作（装/卸 MCP）审批
# full : 全部直行，危险工具也免审批（用户显式选择全开放）
# 矩阵未覆盖的 access（未知类别）按最保守的 approval 处理。
PERMISSION_MATRIX: dict[str, dict[str, str]] = {
    "read": {"read": "allow", "write": "approval", "execute": "approval", "admin": "approval"},
    "write": {"read": "allow", "write": "allow", "execute": "allow", "admin": "approval"},
    "full": {"read": "allow", "write": "allow", "execute": "allow", "admin": "allow"},
}

PERMISSION_DENIED_HINT = (
    "Current permission mode blocks this operation. "
    "Ask the user to switch permission mode (read/write/full) "
    "or propose a read-only alternative."
)

# JSON Schema 简单类型 → Python 校验器（dispatcher 参数校验用，安全设计）
_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


@dataclass(frozen=True)
class ToolResult:
    """单次工具调用的归一化结果（成功与失败同构）。"""

    tool_use_id: str | None
    content: str            # tool_result 块的 content（成功=结果文本，失败=tool_error JSON）
    is_error: bool
    status: str             # ok / error / timeout / denied / interrupted
    approval: str = "auto"  # auto / approved / denied / timeout
    code: str | None = None  # 失败时的错误码（error.type）
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.is_error


ApproveCallback = Callable[[str, Mapping[str, Any]], Awaitable[bool]]
# 审批前通知钩子（AgentLoop 用它发 approval_request SSE）
ApprovalNotifier = Callable[[str, Mapping[str, Any], str], Awaitable[None]]
ToolCall = tuple[str | None, str, dict[str, Any]]  # (tool_use_id, name, input)


class ToolBreaker:
    """按工具名维护的连续失败计数（成功一次清零）。"""

    def __init__(self, limit: int = DEFAULT_BREAKER_LIMIT) -> None:
        self.limit = limit
        self._failures: dict[str, int] = {}

    def is_tripped(self, name: str) -> bool:
        return self._failures.get(name, 0) >= self.limit

    def record_failure(self, name: str) -> int:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        return count

    def record_success(self, name: str) -> None:
        self._failures[name] = 0

    def reset(self, name: str | None = None) -> None:
        """手动解除熔断（name=None 清全部）。供运维/测试恢复用。"""
        if name is None:
            self._failures.clear()
        else:
            self._failures[name] = 0


class ToolDispatcher:
    """AgentLoop 唯一的工具执行入口。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        approve_callback: ApproveCallback | None = None,
        approval_notifier: ApprovalNotifier | None = None,
        breaker_limit: int = DEFAULT_BREAKER_LIMIT,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_backoff_base_s: float = RETRY_BACKOFF_BASE_S,
        breaker: ToolBreaker | None = None,
        permission_mode: str = DEFAULT_PERMISSION_MODE,
    ) -> None:
        self._registry = registry
        self._approve = approve_callback
        self._notify_approval = approval_notifier
        self._breaker = breaker or ToolBreaker(breaker_limit)
        self._default_timeout_s = default_timeout_s
        self._backoff_base = retry_backoff_base_s
        self._permission_mode = self._normalize_mode(permission_mode)
        # parallel_safe=False 的工具串行化锁（跨并发 dispatch_many 共享）
        self._serial_lock = asyncio.Lock()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if mode not in PERMISSION_MODES:
            raise ValueError(
                f"invalid permission mode: {mode!r} (expected one of {PERMISSION_MODES})"
            )
        return mode

    @property
    def permission_mode(self) -> str:
        """当前权限模式（read / write / full）。"""
        return self._permission_mode

    def set_permission_mode(self, mode: str) -> None:
        """热切换权限模式（全局开关，下一个工具调用即按新模式判定）。"""
        self._permission_mode = self._normalize_mode(mode)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        tool_input: Mapping[str, Any],
        tool_use_id: str | None = None,
    ) -> ToolResult:
        """执行单个工具，永不抛异常——一切失败归一化为 is_error 的 ToolResult。"""
        started = time.monotonic()
        try:
            result = await self._dispatch_inner(name, tool_input, tool_use_id)
        except asyncio.CancelledError:
            raise  # 中断：由 AgentLoop 合成 interrupted 结果
        except Exception as exc:  # 兜底：工具代码里的任意异常
            result = self._error_result(
                tool_use_id, code="internal_error",
                message=f"Unhandled error in tool '{name}': {exc}",
                status="error",
            )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return ToolResult(
            tool_use_id=result.tool_use_id,
            content=result.content,
            is_error=result.is_error,
            status=result.status,
            approval=result.approval,
            code=result.code,
            duration_ms=elapsed_ms,
        )

    async def dispatch_many(
        self, calls: Sequence[ToolCall]
    ) -> list[ToolResult]:
        """并行执行多个工具调用（失败隔离）。

        parallel_safe=True 的并发跑；False 的经锁串行。
        return_exceptions=True 确保一个失败不影响其他；漏网异常兜底归一化。
        结果顺序与入参 calls 一一对应。
        """
        if not calls:
            return []

        async def run_one(call: ToolCall) -> ToolResult:
            tool_use_id, name, tool_input = call
            result = await self.dispatch(name, tool_input, tool_use_id)
            return result

        async def guarded(call: ToolCall) -> ToolResult:
            _, name, _ = call
            spec = self._registry.get(name)
            if spec is not None and not spec.parallel_safe:
                async with self._serial_lock:
                    return await run_one(call)
            return await run_one(call)

        gathered = await asyncio.gather(
            *(guarded(call) for call in calls), return_exceptions=True
        )
        results: list[ToolResult] = []
        for call, outcome in zip(calls, gathered):
            if isinstance(outcome, BaseException):
                # dispatch 内部已兜底，这里只剩 CancelledError/KeyboardInterrupt 等
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                results.append(
                    self._error_result(
                        call[0], code="internal_error",
                        message=f"Unhandled dispatcher error: {outcome}",
                        status="error",
                    )
                )
            else:
                results.append(outcome)
        return results

    def reset_breaker(self, name: str | None = None) -> None:
        """解除熔断（name=None 清全部）。"""
        self._breaker.reset(name)

    def set_approval_notifier(self, notifier: ApprovalNotifier | None) -> None:
        """注入审批前通知钩子（dangerous 工具进入审批时回调）。

        AgentLoop.__init__ 用它把 approval_request SSE 接进来——
        dispatcher 本身不感知 SSE 协议。
        """
        self._notify_approval = notifier

    # ------------------------------------------------------------------
    # 内部流水线：熔断检查 → 校验 → 审批 → 重试×超时执行
    # ------------------------------------------------------------------

    async def _dispatch_inner(
        self, name: str, tool_input: Mapping[str, Any], tool_use_id: str | None
    ) -> ToolResult:
        # [2] 未知工具 —— 模型幻觉出不存在的名字
        spec = self._registry.get(name)
        if spec is None:
            return self._error_result(
                tool_use_id, code="unknown_tool",
                message=f"Unknown tool: {name}",
                hint=f"Available tools: {self._registry.names()}",
                status="error",
            )

        # 熔断检查：连续失败 ≥ limit → 拒绝执行（防无限失败循环）
        if self._breaker.is_tripped(name):
            return self._error_result(
                tool_use_id, code="breaker_tripped",
                message=f"Circuit breaker tripped for tool '{name}' "
                        f"after {self._breaker.limit} consecutive failures",
                hint=BREAKER_HINT,
                status="error",
            )

        # [0] 权限判定：按 (模式, access) 矩阵决定放行/审批/拒绝。
        # 未知 access 类别按最保守的 approval；拒绝不计熔断（不是工具的锅）
        action = PERMISSION_MATRIX.get(self._permission_mode, {}).get(
            spec.access if spec.access in ACCESS_LEVELS else "", "approval"
        )
        if action == "deny":
            return self._error_result(
                tool_use_id, code="permission_denied",
                message=f"Tool '{name}' (access={spec.access}) is blocked in "
                        f"'{self._permission_mode}' permission mode",
                hint=PERMISSION_DENIED_HINT,
                status="error",
            )
        approval_required = action == "approval"

        # [1] 参数校验（按 input_schema，轻量校验给出 schema 提示）
        input_dict = dict(tool_input)
        validation_error = self._validate_input(spec, input_dict, tool_use_id)
        if validation_error is not None:
            return validation_error

        # [6] 审批门：矩阵判定 approval 的工具必须先过 approve_callback
        # （原 dangerous 标记的职责由矩阵接管；dangerous 保留为语义说明）
        approval = "auto"
        if approval_required:
            if self._approve is None:
                approval = "auto"  # 未注入审批回调 → 跳过审批
            else:
                if self._notify_approval is not None:
                    await self._notify_approval(name, input_dict, tool_use_id or "")
                try:
                    allowed = await self._approve(name, input_dict)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return self._error_result(
                        tool_use_id, code="approval_channel_error",
                        message=f"Approval callback failed: {exc}",
                        status="error",
                    )
                if not allowed:
                    return ToolResult(
                        tool_use_id=tool_use_id,
                        content=tool_error("approval_denied", DENIED_MESSAGE),
                        is_error=True,
                        status="denied",
                        approval="denied",
                        code="approval_denied",
                    )
                approval = "approved"

        # [3][4][5] 执行：内部重试 × wait_for 超时
        return await self._execute_with_retry(spec, input_dict, tool_use_id, approval)

    async def _execute_with_retry(
        self, spec: ToolSpec, tool_input: dict[str, Any], tool_use_id: str | None, approval: str
    ) -> ToolResult:
        attempts = spec.retryable + 1  # retryable = 内部重试次数，默认 0
        last_error: ToolResult | None = None
        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(self._backoff_base * (2 ** (attempt - 1)))
            result = await self._execute_once(spec, tool_input, tool_use_id, approval)
            if result.ok:
                self._breaker.record_success(spec.name)
                return result
            last_error = result
            if result.code not in ("transient_failure",):
                break  # 只有瞬时失败值得内部重试；永久/超时不重试
        assert last_error is not None
        self._breaker.record_failure(spec.name)
        return last_error

    async def _execute_once(
        self, spec: ToolSpec, tool_input: dict[str, Any], tool_use_id: str | None, approval: str
    ) -> ToolResult:
        timeout_s = spec.timeout_s if spec.timeout_s else self._default_timeout_s
        try:
            raw = await asyncio.wait_for(
                self._call_handler(spec, tool_input), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return self._error_result(
                tool_use_id, code="timeout",
                message=f"Tool '{spec.name}' timed out after {timeout_s}s",
                hint="The tool was force-cancelled. Consider a narrower request.",
                status="timeout",
            )
        except ToolTransientError as exc:
            return self._error_result(
                tool_use_id, code=exc.code, message=str(exc), hint=exc.hint, status="error"
            )
        except ToolError as exc:  # [4] 永久失败 / 参数类工具自报错误
            return self._error_result(
                tool_use_id, code=exc.code, message=str(exc), hint=exc.hint, status="error"
            )
        return ToolResult(
            tool_use_id=tool_use_id,
            content=_normalize_output(raw),
            is_error=False,
            status="ok",
            approval=approval,
        )

    async def _call_handler(self, spec: ToolSpec, tool_input: dict[str, Any]) -> Any:
        outcome = spec.handler(tool_input)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome

    # ------------------------------------------------------------------

    def _validate_input(
        self, spec: ToolSpec, tool_input: dict[str, Any], tool_use_id: str | None
    ) -> ToolResult | None:
        """轻量 schema 校验：必填字段 + JSON 基础类型。返回错误 ToolResult 或 None。"""
        schema = spec.input_schema or {}
        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []

        missing = [key for key in required if key not in tool_input]
        if missing:
            return self._error_result(
                tool_use_id, code="invalid_params",
                message=f"Missing required field(s): {missing}",
                hint=f"Required: {required}; properties: {sorted(properties)}",
                status="error",
            )
        for key, value in tool_input.items():
            expected = (properties.get(key) or {}).get("type")
            if expected is None or key not in properties:
                continue  # 未声明类型的字段不校验（对扩展开放）
            if expected in _JSON_TYPE_CHECKS:
                allowed = _JSON_TYPE_CHECKS[expected]
                if not isinstance(value, allowed) or (expected != "boolean" and isinstance(value, bool)):
                    return self._error_result(
                        tool_use_id, code="invalid_params",
                        message=f"Field '{key}' must be {expected}, got {type(value).__name__}",
                        hint=f"Schema: {json.dumps(properties[key], ensure_ascii=False)}",
                        status="error",
                    )
            elif expected == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return self._error_result(
                        tool_use_id, code="invalid_params",
                        message=f"Field '{key}' must be number, got {type(value).__name__}",
                        status="error",
                    )
            elif expected == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return self._error_result(
                        tool_use_id, code="invalid_params",
                        message=f"Field '{key}' must be integer, got {type(value).__name__}",
                        status="error",
                    )
        return None

    @staticmethod
    def _error_result(
        tool_use_id: str | None,
        *,
        code: str,
        message: str,
        hint: str | None = None,
        status: str,
        approval: str = "auto",
    ) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use_id,
            content=tool_error(code, message, hint),
            is_error=True,
            status=status,
            approval=approval,
            code=code,
        )


def _normalize_output(raw: Any) -> str:
    """工具成功输出 → tool_result.content 字符串。"""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)
