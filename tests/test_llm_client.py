"""LLMClient 单元测试 —— 全部 mock，绝不发起真实 API 调用。

覆盖：流式 delta 回调 + usage 提取、system blocks + cache_control、
thinking adaptive 参数、SDK 类型化异常分类（fatal vs 可重试透传）、
client/client_factory 注入。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import AuthenticationError, PermissionDeniedError, RateLimitError

from app.core.llm_client import (
    LLMClient,
    LLMFatalError,
    UsageData,
    build_system_blocks,
)


# ----------------------------------------------------------------------
# Fake Anthropic client：messages.stream(**kwargs) → async CM stream
# ----------------------------------------------------------------------


class FakeStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self) -> Any:
        return self._final


class FakeMessages:
    def __init__(self, streams: list[FakeStream]) -> None:
        self._streams = streams
        self.kwargs_history: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.kwargs_history.append(kwargs)
        if isinstance(self._streams[0], Exception):
            raise self._streams.pop(0)
        return self._streams.pop(0)


class FakeAnthropic:
    def __init__(self, streams: list[Any]) -> None:
        self.messages = FakeMessages(streams)


def text_delta(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def thinking_delta(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
    )


def final_message(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "content": [{"type": "text", "text": "final text"}],
        "stop_reason": "end_turn",
        "usage": SimpleNamespace(
            input_tokens=100,
            output_tokens=42,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=7,
        ),
        "model": "claude-opus-4-8",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def api_error(cls: type[Exception], status: int) -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


# ----------------------------------------------------------------------


async def test_streaming_deltas_and_usage_extraction() -> None:
    client = LLMClient(
        FakeAnthropic([
            FakeStream(
                events=[text_delta("Hel"), text_delta("lo"), thinking_delta("hmm")],
                final=final_message(),
            )
        ])
    )
    seen: list[str] = []

    async def on_delta(delta: Any) -> None:
        seen.append(f"{delta.kind}:{delta.text}")

    result = await client.run_turn(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        on_delta=on_delta,
    )

    assert seen == ["text:Hel", "text:lo", "thinking:hmm"]
    assert result.stop_reason == "end_turn"
    assert result.content == [{"type": "text", "text": "final text"}]
    assert result.usage == UsageData(
        input_tokens=100, output_tokens=42, cache_read=90, cache_creation=7
    )


async def test_request_params_model_thinking_system_cache_control_tools() -> None:
    fake = FakeAnthropic([FakeStream([], final_message())])
    client = LLMClient(fake, model="claude-opus-4-8", max_tokens=1234)

    await client.run_turn(
        messages=[{"role": "user", "content": []}],
        system="you are an agent",
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        max_tokens=99,  # 显式传参优先
    )

    kwargs = fake.messages.kwargs_history[0]
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["max_tokens"] == 99
    assert kwargs["thinking"] == {"type": "adaptive"}
    # system 冻结为 blocks，最后一个 block 带 cache_control（缓存纪律）
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "you are an agent",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert kwargs["tools"][0]["name"] == "t"


def test_build_system_blocks_variants() -> None:
    blocks = build_system_blocks("s")
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    preexisting = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = build_system_blocks(preexisting)
    assert out[-1]["cache_control"] == {"type": "ephemeral"}
    assert out[0].get("cache_control") is None
    assert build_system_blocks([]) == []


async def test_usage_defaults_when_missing() -> None:
    client = LLMClient(FakeAnthropic([FakeStream([], final_message(usage=None))]))
    result = await client.run_turn(messages=[])
    assert result.usage == UsageData()


# ----------------------------------------------------------------------
# 异常分类（：类型化异常，不做字符串匹配）
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc_cls", "status", "expected_code"),
    [
        (AuthenticationError, 401, "auth_error"),
        (PermissionDeniedError, 403, "permission_denied"),
    ],
)
async def test_fatal_errors_mapped_to_llm_fatal(exc_cls, status, expected_code) -> None:
    client = LLMClient(FakeAnthropic([api_error(exc_cls, status)]))
    with pytest.raises(LLMFatalError) as excinfo:
        await client.run_turn(messages=[])
    assert excinfo.value.code == expected_code


async def test_rate_limit_is_not_fatal_passthrough() -> None:
    """429 交给 SDK 内置重试；耗尽后类型化异常原样上抛，不包装成 fatal。"""
    client = LLMClient(FakeAnthropic([api_error(RateLimitError, 429)]))
    with pytest.raises(RateLimitError):
        await client.run_turn(messages=[])


async def test_client_injection_variants() -> None:
    fake_direct = FakeAnthropic([FakeStream([], final_message())])
    via_client = LLMClient(fake_direct)
    assert await via_client.run_turn(messages=[]) is not None

    fake_factory = FakeAnthropic([FakeStream([], final_message())])
    via_factory = LLMClient(client_factory=lambda: fake_factory)
    assert await via_factory.run_turn(messages=[]) is not None
