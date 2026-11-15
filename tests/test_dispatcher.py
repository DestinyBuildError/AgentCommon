"""ToolDispatcher 单元测试。

覆盖：成功路径、参数错误归一化、超时归一化、未知工具、永久失败附 hint、
内部重试（默认 0 次 / 瞬时重试后成功 / 耗尽）、熔断触发与解除、
并行隔离（一个失败不影响其他）、审批拒绝/放行/自动放行、审批通知钩子。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

from app.core.dispatcher import ToolDispatcher
from app.tools.errors import ToolPermanentError, ToolTransientError, tool_error
from app.tools.registry import ToolRegistry, ToolSpec


def make_registry(specs: list[ToolSpec]) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def spec(
    name: str = "echo",
    handler: Any = None,
    *,
    timeout_s: float = 5.0,
    dangerous: bool = False,
    parallel_safe: bool = True,
    retryable: int = 0,
    input_schema: dict | None = None,
    access: str | None = None,
) -> ToolSpec:
    if handler is None:
        async def handler(tool_input: dict) -> dict:  # type: ignore[misc]
            return {"echo": tool_input}

    # 后审批门由 access 矩阵驱动：dangerous=True 的旧测试语义
    # （"需审批"）对应 execute 类（read 模式下审批）；不传 access 时按此推导
    if access is None:
        access = "execute" if dangerous else "read"

    return ToolSpec(
        name=name,
        description=f"test tool {name}",
        input_schema=input_schema
        or {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=handler,
        timeout_s=timeout_s,
        parallel_safe=parallel_safe,
        dangerous=dangerous,
        access=access,
        retryable=retryable,
    )


def error_payload(result: Any) -> dict:
    return json.loads(result.content)


# ----------------------------------------------------------------------
# 基本路径
# ----------------------------------------------------------------------


async def test_success_path() -> None:
    dispatcher = ToolDispatcher(make_registry([spec()]))
    result = await dispatcher.dispatch("echo", {"q": "hello"}, "tu_1")

    assert result.ok is True
    assert result.is_error is False
    assert result.status == "ok"
    assert result.approval == "auto"
    assert result.tool_use_id == "tu_1"
    payload = json.loads(result.content)
    assert payload == {"echo": {"q": "hello"}}
    assert result.duration_ms >= 0


async def test_string_result_passed_through() -> None:
    registry = make_registry([spec(name="sayer", handler=lambda i: "plain text")])
    result = await ToolDispatcher(registry).dispatch("sayer", {"q": "x"}, "tu_x")
    assert result.content == "plain text"


async def test_unknown_tool_lists_available() -> None:
    registry = make_registry([spec(name="real_tool")])
    result = await ToolDispatcher(registry).dispatch("hallucinated", {}, "tu_1")

    assert result.is_error is True
    assert result.code == "unknown_tool"
    payload = error_payload(result)
    assert payload["error"]["type"] == "unknown_tool"
    assert "real_tool" in payload["error"]["hint"]


# ----------------------------------------------------------------------
# 失败归一化 1：参数不合法
# ----------------------------------------------------------------------


async def test_missing_required_field_is_invalid_params() -> None:
    handler = AsyncMock(return_value={"ok": True})
    dispatcher = ToolDispatcher(make_registry([spec(handler=handler)]))
    result = await dispatcher.dispatch("echo", {}, "tu_1")  # 缺 required q

    assert result.is_error is True
    assert result.code == "invalid_params"
    assert "q" in result.content
    handler.assert_not_awaited()  # 参数非法不执行


async def test_wrong_field_type_is_invalid_params() -> None:
    dispatcher = ToolDispatcher(make_registry([spec()]))
    result = await dispatcher.dispatch("echo", {"q": 123}, "tu_1")
    assert result.is_error is True
    assert result.code == "invalid_params"
    assert "must be string" in result.content


async def test_tool_raised_input_error_normalized() -> None:
    async def handler(tool_input: dict) -> Any:
        raise ToolPermanentError("bad", code="invalid_params", hint="fix it")

    dispatcher = ToolDispatcher(make_registry([spec(handler=handler)]))
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")
    assert result.code == "invalid_params"
    assert error_payload(result)["error"]["hint"] == "fix it"


# ----------------------------------------------------------------------
# 失败归一化 5：超时
# ----------------------------------------------------------------------


async def test_timeout_normalized() -> None:
    async def slow(tool_input: dict) -> Any:
        await asyncio.sleep(5)

    dispatcher = ToolDispatcher(make_registry([spec(handler=slow, timeout_s=0.05)]))
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")

    assert result.is_error is True
    assert result.code == "timeout"
    assert result.status == "timeout"
    assert "timed out after 0.05s" in result.content


# ----------------------------------------------------------------------
# 失败归一化 3/4：瞬时重试、永久失败
# ----------------------------------------------------------------------


async def test_permanent_error_returns_immediately_with_hint() -> None:
    async def missing(tool_input: dict) -> Any:
        raise ToolPermanentError("File not found: x", code="not_found", hint="Directory contains: ['x']")

    dispatcher = ToolDispatcher(make_registry([spec(handler=missing)]))
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")
    assert result.code == "not_found"
    assert error_payload(result)["error"]["hint"].startswith("Directory contains")


async def test_transient_error_default_zero_retries() -> None:
    calls = {"n": 0}

    async def flaky(tool_input: dict) -> Any:
        calls["n"] += 1
        raise ToolTransientError("network blip")

    dispatcher = ToolDispatcher(
        make_registry([spec(handler=flaky)]), retry_backoff_base_s=0
    )
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")
    assert calls["n"] == 1  # 默认 retryable=0：不内部重试
    assert result.code == "transient_failure"


async def test_transient_error_retries_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky(tool_input: dict) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ToolTransientError("network blip")
        return {"attempt": calls["n"]}

    registry = make_registry([spec(handler=flaky, retryable=2)])
    dispatcher = ToolDispatcher(registry, retry_backoff_base_s=0)
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")
    assert calls["n"] == 2
    assert result.ok is True


async def test_unhandled_exception_is_internal_error() -> None:
    async def buggy(tool_input: dict) -> Any:
        raise RuntimeError("segfault")

    dispatcher = ToolDispatcher(make_registry([spec(handler=buggy)]))
    result = await dispatcher.dispatch("echo", {"q": "x"}, "tu_1")
    assert result.is_error is True
    assert result.code == "internal_error"
    assert "segfault" in result.content


def test_tool_error_json_shape() -> None:
    text = tool_error("not_found", "File not found: a", hint="try b")
    assert json.loads(text) == {
        "error": {"type": "not_found", "message": "File not found: a", "hint": "try b"}
    }
    assert json.loads(tool_error("c", "m")) == {"error": {"type": "c", "message": "m"}}


# ----------------------------------------------------------------------
# 熔断器
# ----------------------------------------------------------------------


async def test_breaker_trips_after_consecutive_failures() -> None:
    calls = {"n": 0}

    async def always_fails(tool_input: dict) -> Any:
        calls["n"] += 1
        raise ToolPermanentError("down", code="service_down")

    dispatcher = ToolDispatcher(make_registry([spec(handler=always_fails)]))
    for _ in range(3):
        result = await dispatcher.dispatch("echo", {"q": "x"})
        assert result.code == "service_down"
    assert calls["n"] == 3

    # 第 4 次：熔断触发 → 直接拒绝，不再执行 handler
    tripped = await dispatcher.dispatch("echo", {"q": "x"})
    assert calls["n"] == 3  # handler 未被再次调用
    assert tripped.code == "breaker_tripped"
    assert "Stop retrying this tool" in error_payload(tripped)["error"]["hint"]


async def test_breaker_success_resets_consecutive_count() -> None:
    mode = {"fail": False}

    async def flaky(tool_input: dict) -> Any:
        if mode["fail"]:
            raise ToolPermanentError("down", code="down")
        return "recovered"

    dispatcher = ToolDispatcher(make_registry([spec(handler=flaky)]))

    # 失败两次（连续计数 = 2，未到熔断阈值 3）
    mode["fail"] = True
    for _ in range(2):
        r = await dispatcher.dispatch("echo", {"q": "x"})
        assert r.code == "down"

    # 成功一次 → 连续计数清零
    mode["fail"] = False
    ok = await dispatcher.dispatch("echo", {"q": "x"})
    assert ok.ok is True

    # 再连续失败 2 次（< 3）也不会熔断，handler 仍被真正执行
    mode["fail"] = True
    r1 = await dispatcher.dispatch("echo", {"q": "x"})
    r2 = await dispatcher.dispatch("echo", {"q": "x"})
    assert r1.code == "down" and r2.code == "down"  # 仍是失败，但不是 breaker_tripped


async def test_breaker_reset_reenables_tool() -> None:
    async def always_fails(tool_input: dict) -> Any:
        raise ToolPermanentError("down", code="down")

    dispatcher = ToolDispatcher(make_registry([spec(handler=always_fails)]))
    for _ in range(3):
        await dispatcher.dispatch("echo", {"q": "x"})
    assert (await dispatcher.dispatch("echo", {"q": "x"})).code == "breaker_tripped"

    dispatcher.reset_breaker("echo")
    result = await dispatcher.dispatch("echo", {"q": "x"})
    assert result.code == "down"  # 恢复执行（工具本身仍失败，但不再是熔断错误）


async def test_denied_and_unknown_do_not_count_toward_breaker() -> None:
    async def guarded(tool_input: dict) -> Any:
        return "ran"

    dispatcher = ToolDispatcher(
        make_registry([spec(name="risky", handler=guarded, dangerous=True)]),
        approve_callback=AsyncMock(return_value=False),
    )
    for _ in range(5):
        result = await dispatcher.dispatch("risky", {"q": "x"})
        assert result.code == "approval_denied"  # 不是 breaker_tripped
    # 换正常工具不受影响；被拒工具下次（若放行）仍可执行
    result = await dispatcher.dispatch("unknown_xyz", {})
    assert result.code == "unknown_tool"


# ----------------------------------------------------------------------
# 并行隔离
# ----------------------------------------------------------------------


async def test_parallel_isolation_one_failure_does_not_affect_others() -> None:
    async def ok_tool(tool_input: dict) -> Any:
        await asyncio.sleep(0.01)
        return {"fine": True}

    async def bad_tool(tool_input: dict) -> Any:
        raise RuntimeError("exploded")

    registry = make_registry([
        spec(name="good", handler=ok_tool),
        spec(name="bad", handler=bad_tool),
    ])
    dispatcher = ToolDispatcher(registry)
    results = await dispatcher.dispatch_many([
        ("tu_1", "good", {"q": "a"}),
        ("tu_2", "bad", {"q": "b"}),
        ("tu_3", "good", {"q": "c"}),
    ])

    assert len(results) == 3
    assert results[0].ok is True  # 失败不影响其他
    assert results[1].is_error is True and results[1].code == "internal_error"
    assert results[2].ok is True
    assert [r.tool_use_id for r in results] == ["tu_1", "tu_2", "tu_3"]  # 顺序保持


async def test_dispatch_many_empty() -> None:
    assert await ToolDispatcher(make_registry([])).dispatch_many([]) == []


async def test_non_parallel_safe_tools_serialize() -> None:
    order: list[str] = []
    active = {"current": None}

    async def exclusive(tool_input: dict) -> Any:
        assert active["current"] is None, "parallel-unsafe tool overlapped!"
        active["current"] = tool_input["q"]
        await asyncio.sleep(0.01)
        active["current"] = None
        order.append(tool_input["q"])
        return tool_input["q"]

    registry = make_registry([spec(name="lonely", handler=exclusive, parallel_safe=False)])
    dispatcher = ToolDispatcher(registry)
    results = await dispatcher.dispatch_many([
        ("t1", "lonely", {"q": "a"}),
        ("t2", "lonely", {"q": "b"}),
    ])
    assert all(r.ok for r in results)
    assert order == ["a", "b"]  # 严格串行


# ----------------------------------------------------------------------
# 审批门
# ----------------------------------------------------------------------


async def test_approval_denied_returns_do_not_retry_message() -> None:
    handler = AsyncMock(return_value={"done": 1})
    approve = AsyncMock(return_value=False)
    dispatcher = ToolDispatcher(
        make_registry([spec(name="risky", handler=handler, dangerous=True)]),
        approve_callback=approve,
    )
    result = await dispatcher.dispatch("risky", {"q": "x"}, "tu_9")

    approve.assert_awaited_once_with("risky", {"q": "x"})
    handler.assert_not_awaited()  # 拒绝后不执行
    assert result.is_error is True
    assert result.status == "denied"
    assert result.approval == "denied"
    assert result.code == "approval_denied"
    assert json.loads(result.content)["error"]["message"] == (
        "user denied, do NOT retry, propose alternative"
    )


async def test_approval_approved_executes() -> None:
    handler = AsyncMock(return_value={"done": 1})
    dispatcher = ToolDispatcher(
        make_registry([spec(name="risky", handler=handler, dangerous=True)]),
        approve_callback=AsyncMock(return_value=True),
    )
    result = await dispatcher.dispatch("risky", {"q": "x"}, "tu_9")
    handler.assert_awaited_once()
    assert result.ok is True
    assert result.approval == "approved"
    assert result.status == "ok"


async def test_dangerous_tool_without_callback_auto_allows() -> None:
    handler = AsyncMock(return_value={"done": 1})
    dispatcher = ToolDispatcher(
        make_registry([spec(name="risky", handler=handler, dangerous=True)])
    )
    result = await dispatcher.dispatch("risky", {"q": "x"})
    handler.assert_awaited_once()
    assert result.approval == "auto"


async def test_non_dangerous_tool_skips_approval_callback() -> None:
    handler = AsyncMock(return_value={"done": 1})
    approve = AsyncMock(return_value=False)  # 即便回调总拒绝
    dispatcher = ToolDispatcher(
        make_registry([spec(name="safe", handler=handler)]),  # dangerous=False
        approve_callback=approve,
    )
    result = await dispatcher.dispatch("safe", {"q": "x"})
    approve.assert_not_awaited()
    assert result.ok is True


async def test_approval_notifier_hook_called_before_callback() -> None:
    notifier_calls: list[tuple[str, str]] = []

    async def notifier(tool_name: str, tool_input: Any, tool_use_id: str) -> None:
        notifier_calls.append((tool_name, tool_use_id))

    async def guard(tool_name: str, tool_input: Any) -> bool:
        return True

    dispatcher = ToolDispatcher(
        make_registry([spec(name="risky", handler=AsyncMock(return_value=1), dangerous=True)]),
        approve_callback=guard,
        approval_notifier=notifier,
    )
    await dispatcher.dispatch("risky", {"q": "x"}, "tu_42")
    assert notifier_calls == [("risky", "tu_42")]
