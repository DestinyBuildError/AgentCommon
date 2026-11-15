"""上下文管理 —— TokenLedger + Compactor。

- 每轮从 response.usage 累计 input/output/cache token
- 接近阈值（150K）时 should_compact() 返回 True
- compact() 经 Compactor 调 anthropic beta（compact-2026-01-12）；
  **关键坑**：必须把完整 response.content（含 compaction 块）回传进
  对话历史，只回传文本会静默丢失压缩状态。真实 API 联调列为遗留项
  （无真实 key，本层用注入 client 单测调用路径与回传语义）。

上下文规模估算口径：compaction 是否需要取决于「下一次请求要发多大」，
即最近一轮 API 返回的 input_tokens（API 报告的就是整个前缀的规模，
含 cache_read 部分）。因此 context_estimate 取最近一轮 input_tokens
的滚动值，而非跨轮累加值——累加值只用于会话级用量统计（已落在
sessions 表，职责不在这里）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.llm_client import UsageData

COMPACTION_THRESHOLD = 150_000
# anthropic beta flag：上下文压缩。真实联调遗留项。
COMPACTION_BETA = "compact-2026-01-12"


class TokenLedger:
    """会话级 token 台账：累计用量 + 上下文规模估算 + compaction 阈值判断。"""

    def __init__(self, threshold: int = COMPACTION_THRESHOLD) -> None:
        self.threshold = threshold
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.turns = 0
        # 最近一轮 input_tokens：API 每轮报告完整前缀规模，即当前上下文近似值
        self.context_estimate = 0

    def add(self, usage: UsageData) -> None:
        """累加一轮用量，并滚动更新上下文规模估算。"""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read_tokens += usage.cache_read
        self.total_cache_creation_tokens += usage.cache_creation
        self.context_estimate = usage.input_tokens
        self.turns += 1

    def should_compact(self, threshold: int | None = None) -> bool:
        """上下文估算超过阈值（默认 150K）→ 需要 compaction。"""
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
        """执行 compaction：调 anthropic beta（compact-2026-01-12）。

        返回值是**完整 response.content**（含 compaction 块）——调用方
        （AgentLoop）必须把它原样回传进对话历史（追加为 assistant 消息），
        只回传文本会静默丢失压缩状态。未配置 compactor 时抛 CompactionError。
        """
        if compactor is None:
            raise CompactionError("no compactor configured")
        return await compactor.compact(
            messages=messages, system=system, tools=tools, model=model
        )


class CompactionError(Exception):
    """compaction 失败（调用出错 / 响应为空）。调用方降级：不阻塞主循环。"""


class Compactor:
    """上下文压缩器：anthropic beta compact-2026-01-12。

    client 由构造注入（AsyncAnthropic 或测试 mock），本层绝不自建连接；
    真实 API 联调列为遗留项——SDK 的 beta compact 入参形态若与实现有出入，
    只需调整 _invoke_compact 一处。
    """

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
        """请求压缩，返回完整 response.content（含 compaction 块）。

        响应为空视为失败（压缩静默丢失比报错更危险）。
        """
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
        """beta compact 端点调用（真实联调遗留项的唯一切换点）。"""
        return await self._client.beta.messages.compact(**kwargs)
