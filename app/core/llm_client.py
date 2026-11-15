"""LLM 客户端层 —— Anthropic SDK 封装。

职责：
- 流式调用（client.messages.stream + get_final_message），逐 delta 回调
- 异常分类用 SDK 类型化异常，不做字符串匹配：
  * RateLimitError / APIStatusError(5xx) / APIConnectionError / APITimeoutError
    → 交给 SDK 内置指数退避（max_retries=3，本层不自写重试），
      重试耗尽后类型化异常原样上抛，由 AgentLoop 归一化为 error 事件
  * AuthenticationError / PermissionDeniedError / NotFoundError / BadRequestError
    → 映射为 LLMFatalError（认证/计费/请求本身非法，重试无意义），
      AgentLoop 捕获后终止循环报告用户，不进重试通道
- 从 response.usage 提取标准化 UsageData

构造函数接受可注入的 client / client_factory —— 测试用 mock 替换，
绝不发起真实 API 调用。
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 16000
SDK_MAX_RETRIES = 3  # 第①层：SDK 内置退避重试，不自己重写


class LLMFatalError(Exception):
    """致命 LLM 错误（认证/权限/请求非法）：终止循环报告用户，不自动重试。

    code 用于 SSE error 事件的错误码：
      auth_error / permission_denied / model_not_found / invalid_request
    """

    def __init__(self, message: str, *, code: str = "llm_fatal") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UsageData:
    """单轮标准化用量（列名与 messages 表 / sessions 表对齐）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    def as_dict(self) -> dict[str, int]:
        """Anthropic SDK 风格键名——Repository.append_message 的 usage 入参。"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read,
            "cache_creation_input_tokens": self.cache_creation,
        }


@dataclass(frozen=True)
class StreamDelta:
    """流式中间增量（协议无关）。

    kind: "text" → SSE text_delta；"thinking" → SSE thinking_delta。
    """

    kind: Literal["text", "thinking"]
    text: str


@dataclass
class LLMTurnResult:
    """单轮最终结果（流结束后的完整消息）。"""

    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: UsageData = field(default_factory=UsageData)
    model: str = DEFAULT_MODEL


DeltaCallback = Callable[[StreamDelta], Awaitable[None]]

_FATAL_EXCEPTION_CODES: tuple[type[Exception], ...] = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
)


def fatal_code_for(exc: Exception) -> str | None:
    """SDK 类型化异常 → fatal 错误码；非 fatal 返回 None（：不做字符串匹配）。"""
    if isinstance(exc, AuthenticationError):
        return "auth_error"
    if isinstance(exc, PermissionDeniedError):
        return "permission_denied"
    if isinstance(exc, NotFoundError):
        return "model_not_found"
    if isinstance(exc, BadRequestError):
        return "invalid_request"
    return None


def is_retryable_llm_error(exc: Exception) -> bool:
    """429 / 5xx / 连接错误：SDK 已内置退避重试；耗尽后仍属可重试类（非 fatal）。"""
    return isinstance(
        exc, (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)
    )


def build_system_blocks(system: str | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """system → blocks + cache_control（缓存纪律：system prompt 冻结可缓存）。"""
    if isinstance(system, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    else:
        blocks = [dict(block) for block in system]
    if not blocks:
        return []
    # cache_control 标在最后一个 block 上（前缀匹配整段 system）
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return blocks


class LLMClient:
    """AsyncAnthropic 薄封装。注入 client 或 client_factory 即可 mock。"""

    def __init__(
        self,
        client: Any | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = SDK_MAX_RETRIES,
    ) -> None:
        if client is not None:
            self._client = client
        elif client_factory is not None:
            self._client = client_factory()
        else:
            self._client = AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY") or "",
                max_retries=max_retries,
            )
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------

    async def run_turn(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system: str | Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMTurnResult:
        """流式执行一轮对话，返回最终消息。

        on_delta：每个 text/thinking delta 回调一次（AgentLoop 转成 SSEEvent）。
        异常分类见模块 docstring。
        """
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": [dict(m) for m in messages],
            "thinking": {"type": "adaptive"},  # /
        }
        if system:
            kwargs["system"] = build_system_blocks(system)
        if tools:
            kwargs["tools"] = [dict(t) for t in tools]

        try:
            deltas: list[StreamDelta] = []
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        d = StreamDelta(kind="text", text=delta.text)
                    elif delta.type == "thinking_delta":
                        d = StreamDelta(kind="thinking", text=delta.thinking)
                    else:  # input_json_delta / signature_delta 等不外发
                        continue
                    deltas.append(d)
                    if on_delta is not None:
                        await on_delta(d)
                final = await stream.get_final_message()
        except _FATAL_EXCEPTION_CODES as exc:
            raise LLMFatalError(str(exc), code=fatal_code_for(exc) or "llm_fatal") from exc
        # RateLimitError / APIStatusError / 连接类：SDK 已重试耗尽，原样上抛
        # （类型化异常，AgentLoop 据此归一化 error 事件——不做字符串匹配）

        usage_raw = getattr(final, "usage", None)
        usage = _extract_usage(usage_raw)
        content = [dict(block) for block in getattr(final, "content", []) or []]
        return LLMTurnResult(
            content=content,
            stop_reason=getattr(final, "stop_reason", None),
            usage=usage,
            model=getattr(final, "model", None) or kwargs["model"],
        )


def _extract_usage(usage_raw: Any) -> UsageData:
    if usage_raw is None:
        return UsageData()
    return UsageData(
        input_tokens=int(getattr(usage_raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage_raw, "output_tokens", 0) or 0),
        cache_read=int(getattr(usage_raw, "cache_read_input_tokens", 0) or 0),
        cache_creation=int(getattr(usage_raw, "cache_creation_input_tokens", 0) or 0),
    )
