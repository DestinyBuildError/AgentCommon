
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
SDK_MAX_RETRIES = 3


class LLMFatalError(Exception):

    def __init__(self, message: str, *, code: str = "llm_fatal") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UsageData:

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read,
            "cache_creation_input_tokens": self.cache_creation,
        }


@dataclass(frozen=True)
class StreamDelta:

    kind: Literal["text", "thinking"]
    text: str


@dataclass
class LLMTurnResult:

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
    return isinstance(
        exc, (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)
    )


def build_system_blocks(system: str | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(system, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    else:
        blocks = [dict(block) for block in system]
    if not blocks:
        return []
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return blocks


class LLMClient:

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
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": [dict(m) for m in messages],
            "thinking": {"type": "adaptive"},
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
                    else:
                        continue
                    deltas.append(d)
                    if on_delta is not None:
                        await on_delta(d)
                final = await stream.get_final_message()
        except _FATAL_EXCEPTION_CODES as exc:
            raise LLMFatalError(str(exc), code=fatal_code_for(exc) or "llm_fatal") from exc

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
