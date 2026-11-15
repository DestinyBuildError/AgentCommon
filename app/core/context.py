
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.llm_client import UsageData

COMPACTION_THRESHOLD = 150_000
COMPACTION_BETA = "compact-2026-01-12"


class TokenLedger:

    def __init__(self, threshold: int = COMPACTION_THRESHOLD) -> None:
        self.threshold = threshold
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.turns = 0
        self.context_estimate = 0

    def add(self, usage: UsageData) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read_tokens += usage.cache_read
        self.total_cache_creation_tokens += usage.cache_creation
        self.context_estimate = usage.input_tokens
        self.turns += 1

    def should_compact(self, threshold: int | None = None) -> bool:
        limit = self.threshold if threshold is None else threshold
        return self.context_estimate >= limit

    async def compact(
        self,
        compactor: Compactor,
        *,
        messages: Sequence[dict[str, Any]],
        system: Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        if compactor is None:
            raise CompactionError("no compactor configured")
        return await compactor.compact(
            messages=messages, system=system, tools=tools, model=model
        )


class CompactionError(Exception):
    pass


class Compactor:

    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_tokens: int = 8_192,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def compact(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        system: Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": self._max_tokens,
            "messages": [dict(m) for m in messages],
            "betas": [COMPACTION_BETA],
        }
        if system:
            kwargs["system"] = [dict(b) for b in system]
        if tools:
            kwargs["tools"] = [dict(t) for t in tools]
        response = await self._invoke_compact(kwargs)
        content = [dict(block) for block in getattr(response, "content", []) or []]
        if not content:
            raise CompactionError("compaction response has no content blocks")
        return content

    async def _invoke_compact(self, kwargs: dict[str, Any]) -> Any:
        return await self._client.beta.messages.compact(**kwargs)
