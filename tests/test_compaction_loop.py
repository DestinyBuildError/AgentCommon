
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.agent_loop import AgentLoop
from app.core.context import COMPACTION_BETA, Compactor
from app.core.dispatcher import ToolDispatcher
from app.core.llm_client import LLMTurnResult, UsageData
from app.db.database import build_engine
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import SSEEvent, SSEEventType
from app.tools.registry import ToolRegistry, ToolSpec


@pytest.fixture
async def db_session() -> AsyncSession:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def repo(db_session: AsyncSession) -> Repository:
    return Repository(db_session)


@pytest.fixture
def registry() -> ToolRegistry:
    registry_ = ToolRegistry()

    async def echo(tool_input: dict) -> dict:
        return {"echoed": tool_input.get("text", "")}

    registry_.register(
        ToolSpec(
            name="echo",
            description="echo text back",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=echo,
            timeout_s=5,
        )
    )
    return registry_


class ScriptedLLM:
    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def run_turn(self, *, messages, system=None, tools=None, model=None,
                       max_tokens=None, on_delta=None) -> LLMTurnResult:
        self.calls.append({"messages": deepcopy(messages)})
        turn = self.turns.pop(0)
        return turn


class MockCompactClient:

    def __init__(self, content: list[dict[str, Any]]) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    @property
    def beta(self) -> Any:
        outer = self

        class _Messages:
            async def compact(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)

                class _Resp:
                    content = [dict(b) for b in outer._content]

                return _Resp()

        class _Beta:
            messages = _Messages()

        return _Beta()


COMPACTION_CONTENT = [
    {"type": "compaction", "summary": "history compacted"},
    {"type": "text", "text": "Compacted summary of the conversation so far."},
]


def big_usage(input_tokens: int) -> UsageData:
    return UsageData(input_tokens=input_tokens, output_tokens=10, cache_read=0, cache_creation=0)


def make_loop(repo: Repository, registry: ToolRegistry, llm: ScriptedLLM, **kwargs: Any) -> AgentLoop:
    dispatcher = ToolDispatcher(registry, retry_backoff_base_s=0)
    return AgentLoop(
        llm=llm,
        repo=repo,
        registry=registry,
        dispatcher=dispatcher,
        system_prompt="test",
        model="claude-opus-4-8",
        **kwargs,
    )


async def test_compaction_triggers_and_full_content_round_tripped(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM([
        LLMTurnResult(
            content=[{"type": "text", "text": "final"}],
            stop_reason="end_turn",
            usage=big_usage(200),
        ),
    ])
    client = MockCompactClient(COMPACTION_CONTENT)
    loop = make_loop(
        repo, registry, llm,
        compaction_threshold=100, compactor=Compactor(client),
    )
    outcome = await loop.run(session.id, "hello")

    assert outcome.ok is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["betas"] == [COMPACTION_BETA]
    assert any(
        b.get("text") == "hello"
        for m in call["messages"] for b in m["content"]
    )
    messages = await repo.get_messages(session.id)
    contents = [m.content for m in messages]
    assert COMPACTION_CONTENT in contents
    assert contents.index(COMPACTION_CONTENT) == 1
    assert contents[-1] == [{"type": "text", "text": "final"}]


async def test_compaction_once_per_run(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM([
        LLMTurnResult(
            content=[{
                "type": "tool_use", "id": "tu_1", "name": "echo",
                "input": {"text": "hi"},
            }],
            stop_reason="tool_use",
            usage=big_usage(300),
        ),
        LLMTurnResult(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
            usage=big_usage(400),
        ),
    ])
    client = MockCompactClient(COMPACTION_CONTENT)
    loop = make_loop(repo, registry, llm, compaction_threshold=100, compactor=Compactor(client))
    outcome = await loop.run(session.id, "go")
    assert outcome.ok is True
    assert len(client.calls) == 1


async def test_compaction_failure_degrades_loop(
    repo: Repository, registry: ToolRegistry
) -> None:

    class ExplodingCompactor:
        async def compact(self, **kwargs: Any) -> Any:
            raise RuntimeError("compact endpoint unavailable")

    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM([
        LLMTurnResult(
            content=[{"type": "text", "text": "still fine"}],
            stop_reason="end_turn",
            usage=big_usage(500),
        ),
    ])
    loop = make_loop(
        repo, registry, llm,
        compaction_threshold=100, compactor=ExplodingCompactor(),
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop._event_sink = sink
    outcome = await loop.run(session.id, "hello")
    assert outcome.ok is True
    assert outcome.stop_reason == "end_turn"
    messages = await repo.get_messages(session.id)
    assert messages[-1].content == [{"type": "text", "text": "still fine"}]


async def test_no_compactor_keeps_baseline_behavior(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM([
        LLMTurnResult(
            content=[{"type": "text", "text": "no compaction"}],
            stop_reason="end_turn",
            usage=big_usage(999),
        ),
    ])
    loop = make_loop(repo, registry, llm, compaction_threshold=100)
    outcome = await loop.run(session.id, "hello")
    assert outcome.ok is True
    messages = await repo.get_messages(session.id)
    assert len(messages) == 2


async def test_turn_complete_carries_real_cost_and_session_accumulates(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM([
        LLMTurnResult(
            content=[{
                "type": "tool_use", "id": "tu_c", "name": "echo",
                "input": {"text": "x"},
            }],
            stop_reason="tool_use",
            usage=UsageData(input_tokens=1000, output_tokens=100),
        ),
        LLMTurnResult(
            content=[{"type": "text", "text": "b"}],
            stop_reason="end_turn",
            usage=UsageData(input_tokens=2000, output_tokens=100),
        ),
    ])
    loop = make_loop(repo, registry, llm)
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop._event_sink = sink
    outcome = await loop.run(session.id, "hello")

    completes = [e for e in events if e.type is SSEEventType.TURN_COMPLETE]
    assert len(completes) == 1
    assert completes[0].data["cost_usd"] == pytest.approx(0.0075 + 0.0125)

    assert outcome.cost_usd == pytest.approx(0.02)
    updated = await repo.get_session(session.id)
    assert updated.total_cost_usd == pytest.approx(0.02)
    assert updated.total_input_tokens == 3000
