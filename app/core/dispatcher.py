
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

RETRY_BACKOFF_BASE_S = 0.1

PERMISSION_MODES = ("read", "write", "full")
DEFAULT_PERMISSION_MODE = "read"

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

_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


@dataclass(frozen=True)
class ToolResult:

    tool_use_id: str | None
    content: str
    is_error: bool
    status: str
    approval: str = "auto"
    code: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.is_error


ApproveCallback = Callable[[str, Mapping[str, Any]], Awaitable[bool]]
ApprovalNotifier = Callable[[str, Mapping[str, Any], str], Awaitable[None]]
ToolCall = tuple[str | None, str, dict[str, Any]]


class ToolBreaker:

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
        if name is None:
            self._failures.clear()
        else:
            self._failures[name] = 0


class ToolDispatcher:

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
        return self._permission_mode

    def set_permission_mode(self, mode: str) -> None:
        self._permission_mode = self._normalize_mode(mode)


    async def dispatch(
        self,
        name: str,
        tool_input: Mapping[str, Any],
        tool_use_id: str | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        try:
            result = await self._dispatch_inner(name, tool_input, tool_use_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
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
        self._breaker.reset(name)

    def set_approval_notifier(self, notifier: ApprovalNotifier | None) -> None:
        self._notify_approval = notifier


    async def _dispatch_inner(
        self, name: str, tool_input: Mapping[str, Any], tool_use_id: str | None
    ) -> ToolResult:
        spec = self._registry.get(name)
        if spec is None:
            return self._error_result(
                tool_use_id, code="unknown_tool",
                message=f"Unknown tool: {name}",
                hint=f"Available tools: {self._registry.names()}",
                status="error",
            )

        if self._breaker.is_tripped(name):
            return self._error_result(
                tool_use_id, code="breaker_tripped",
                message=f"Circuit breaker tripped for tool '{name}' "
                        f"after {self._breaker.limit} consecutive failures",
                hint=BREAKER_HINT,
                status="error",
            )

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

        input_dict = dict(tool_input)
        validation_error = self._validate_input(spec, input_dict, tool_use_id)
        if validation_error is not None:
            return validation_error

        approval = "auto"
        if approval_required:
            if self._approve is None:
                approval = "auto"
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

        return await self._execute_with_retry(spec, input_dict, tool_use_id, approval)

    async def _execute_with_retry(
        self, spec: ToolSpec, tool_input: dict[str, Any], tool_use_id: str | None, approval: str
    ) -> ToolResult:
        attempts = spec.retryable + 1
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
                break
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
        except ToolError as exc:
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


    def _validate_input(
        self, spec: ToolSpec, tool_input: dict[str, Any], tool_use_id: str | None
    ) -> ToolResult | None:
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
                continue
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
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)
