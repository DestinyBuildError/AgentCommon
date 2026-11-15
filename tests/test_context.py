"""TokenLedger 单元测试（上下文管理）。

累计用量、上下文规模估算口径（最近一轮 input）、compaction 阈值判断、
Compactor（anthropic beta compact-2026-01-12）调用路径与
"完整 response.content 回传"语义——全部 mock client，不发真实请求。
"""

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
    """上下文规模 = 最近一轮 input_tokens（API 报告完整前缀规模），非累加值。"""
    ledger = TokenLedger()
    ledger.add(UsageData(input_tokens=50_000))
    assert ledger.context_estimate == 50_000
    ledger.add(UsageData(input_tokens=80_000))
    assert ledger.context_estimate == 80_000  # 滚动覆盖，不累加


def test_should_compact_threshold() -> None:
    ledger = TokenLedger(threshold=1000)
    ledger.add(UsageData(input_tokens=999))
    assert ledger.should_compact() is False
    ledger.add(UsageData(input_tokens=1000))
    assert ledger.should_compact() is True  # >= 阈值即触发

    # 显式传参覆盖实例阈值
    assert ledger.should_compact(threshold=2000) is False


def test_default_threshold_is_150k() -> None:
    assert COMPACTION_THRESHOLD == 150_000
    ledger = TokenLedger()
    assert ledger.threshold == COMPACTION_THRESHOLD
    ledger.add(UsageData(input_tokens=149_999))
    assert ledger.should_compact() is False
    ledger.add(UsageData(input_tokens=150_000))
    assert ledger.should_compact() is True


# ----------------------------------------------------------------------
# Compactor：mock client 验证调用路径与回传语义
# ----------------------------------------------------------------------


class MockCompactResponse:
    def __init__(self, content: list[dict[str, Any]]) -> None:
        self.content = content


class MockCompactClient:
    """记录 beta.messages.compact 调用入参并回放预设响应/异常。"""

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


# compaction 块样例：真实 API 的压缩响应含特殊块类型，只回传文本会
# 静默丢失压缩状态（关键坑）
COMPACTION_CONTENT = [
    {"type": "compaction", "data": {"summary": "earlier conversation compacted"}},
    {"type": "text", "text": "Earlier context summarized: user asked about pricing."},
]


@pytest.mark.asyncio
async def test_compactor_calls_beta_compact_and_returns_full_content() -> None:
    """调用路径：betas 带 compact-2026-01-12；返回完整 content（含 compaction 块）。"""
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
    assert content == COMPACTION_CONTENT  # 完整回传语义：块原样透传
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
    """空响应（无 content 块）→ CompactionError：静默丢压缩状态比报错更危险。"""
    client = MockCompactClient(MockCompactResponse(content=[]))
    compactor = Compactor(client)
    with pytest.raises(CompactionError):
        await compactor.compact(messages=[{"role": "user", "content": []}])


@pytest.mark.asyncio
async def test_compactor_client_error_propagates() -> None:
    """client 抛错原样上抛（AgentLoop 捕获后降级，不阻塞主循环）。"""
    client = MockCompactClient(RuntimeError("boom"))
    compactor = Compactor(client)
    with pytest.raises(RuntimeError):
        await compactor.compact(messages=[{"role": "user", "content": []}])


@pytest.mark.asyncio
async def test_ledger_compact_requires_compactor() -> None:
    ledger = TokenLedger()
    with pytest.raises(CompactionError):
        await ledger.compact(
            None, messages=[{"role": "user", "content": []}]  # type: ignore[arg-type]
        )
    client = MockCompactClient(MockCompactResponse(COMPACTION_CONTENT))
    content = await ledger.compact(
        Compactor(client), messages=[{"role": "user", "content": []}]
    )
    assert content == COMPACTION_CONTENT  # ledger 委托透传完整 content
