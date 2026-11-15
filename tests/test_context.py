
from __future__ import annotations

from typing import Any

import pytest

from app.core.context import (
    COMPACTION_BETA,
    COMPACTION_THRESHOLD,
    CompactionError,
    Compactor,
    TokenLedger,
)
from app.core.llm_client import UsageData


def test_ledger_accumulates_usage() -> None:
    ledger = TokenLedger()
    ledger.add(UsageData(input_tokens=100, output_tokens=20, cache_read=50, cache_creation=5))
    ledger.add(UsageData(input_tokens=120, output_tokens=30, cache_read=60, cache_creation=0))
    assert ledger.total_input_tokens == 220
    assert ledger.total_output_tokens == 50
    assert ledger.total_cache_read_tokens == 110
    assert ledger.total_cache_creation_tokens == 5
    assert ledger.turns == 2


def test_context_estimate_tracks_last_turn_input() -> None:
    ledger = TokenLedger()
    ledger.add(UsageData(input_tokens=50_000))
    assert ledger.context_estimate == 50_000
    ledger.add(UsageData(input_tokens=80_000))
    assert ledger.context_estimate == 80_000


def test_should_compact_threshold() -> None:
    ledger = TokenLedger(threshold=1000)
    ledger.add(UsageData(input_tokens=999))
    assert ledger.should_compact() is False
    ledger.add(UsageData(input_tokens=1000))
    assert ledger.should_compact() is True

    assert ledger.should_compact(threshold=2000) is False


def test_default_threshold_is_150k() -> None:
    assert COMPACTION_THRESHOLD == 150_000
    ledger = TokenLedger()
    assert ledger.threshold == COMPACTION_THRESHOLD
    ledger.add(UsageData(input_tokens=149_999))
    assert ledger.should_compact() is False
    ledger.add(UsageData(input_tokens=150_000))
    assert ledger.should_compact() is True


class MockCompactResponse:
    def __init__(self, content: list[dict[str, Any]]) -> None:
        self.content = content


class MockCompactClient:

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def _compact(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    @property
    def beta(self) -> Any:
        outer = self

        class _BetaMessages:
            async def compact(self, **kwargs: Any) -> Any:
                return await outer._compact(**kwargs)

        class _Beta:
            messages = _BetaMessages()

        return _Beta()


COMPACTION_CONTENT = [
    {"type": "compaction", "data": {"summary": "earlier conversation compacted"}},
    {"type": "text", "text": "Earlier context summarized: user asked about pricing."},
]


@pytest.mark.asyncio
async def test_compactor_calls_beta_compact_and_returns_full_content() -> None:
    client = MockCompactClient(MockCompactResponse(COMPACTION_CONTENT))
    compactor = Compactor(client, model="claude-opus-4-8", max_tokens=4096)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    content = await compactor.compact(
        messages=messages,
        system=[{"type": "text", "text": "sys"}],
        tools=[{"name": "echo", "input_schema": {"type": "object"}}],
        model="claude-opus-4-8",
    )
    assert content == COMPACTION_CONTENT
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["betas"] == [COMPACTION_BETA]
    assert COMPACTION_BETA == "compact-2026-01-12"
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 4096
    assert call["messages"] == messages
    assert call["system"] == [{"type": "text", "text": "sys"}]
    assert call["tools"] == [{"name": "echo", "input_schema": {"type": "object"}}]


@pytest.mark.asyncio
async def test_compactor_empty_response_raises_compaction_error() -> None:
    client = MockCompactClient(MockCompactResponse(content=[]))
    compactor = Compactor(client)
    with pytest.raises(CompactionError):
        await compactor.compact(messages=[{"role": "user", "content": []}])


@pytest.mark.asyncio
async def test_compactor_client_error_propagates() -> None:
    client = MockCompactClient(RuntimeError("boom"))
    compactor = Compactor(client)
    with pytest.raises(RuntimeError):
        await compactor.compact(messages=[{"role": "user", "content": []}])


@pytest.mark.asyncio
async def test_ledger_compact_requires_compactor() -> None:
    ledger = TokenLedger()
    with pytest.raises(CompactionError):
        await ledger.compact(
            None, messages=[{"role": "user", "content": []}]
        )
    client = MockCompactClient(MockCompactResponse(COMPACTION_CONTENT))
    content = await ledger.compact(
        Compactor(client), messages=[{"role": "user", "content": []}]
    )
    assert content == COMPACTION_CONTENT
