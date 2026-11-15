
from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from unittest.mock import AsyncMock

from app.core.agent_loop import AgentLoop, SessionNotFoundError, _merge_consecutive
from app.core.context import TokenLedger
from app.core.dispatcher import ToolDispatcher
from app.core.llm_client import LLMFatalError, LLMTurnResult, StreamDelta, UsageData
from app.core.repair import needs_repair
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
    registry = ToolRegistry()

    async def echo(tool_input: dict) -> dict:
        return {"echoed": tool_input.get("text", "")}

    async def risky(tool_input: dict) -> dict:
        return {"blew_up": True}

    registry.register(
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
    registry.register(
        ToolSpec(
            name="risky",
            description="dangerous test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=risky,
            timeout_s=5,
            dangerous=True,
            access="execute",
        )
    )
    return registry


class ScriptedLLM:

    def __init__(self, turns: list[Any], deltas: list[list[StreamDelta]] | None = None) -> None:
        self.turns = list(turns)
        self.deltas = list(deltas or [])
        self.calls: list[dict[str, Any]] = []

    async def run_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        system: Any = None,
        tools: Any = None,
        model: str | None = None,
        max_tokens: int | None = None,
        on_delta: Any = None,
    ) -> LLMTurnResult:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "system": deepcopy(system),
                "tools": deepcopy(tools),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        if self.deltas:
            for delta in self.deltas.pop(0):
                if on_delta is not None:
                    await on_delta(delta)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def build_loop(
    repo: Repository,
    registry: ToolRegistry,
    llm: ScriptedLLM,
    *,
    event_sink=None,
    max_iterations: int = 30,
    max_tokens: int = 16000,
    approve_callback: Any = None,
) -> AgentLoop:
    dispatcher = ToolDispatcher(
        registry, approve_callback=approve_callback, retry_backoff_base_s=0
    )
    return AgentLoop(
        llm=llm,
        repo=repo,
        registry=registry,
        dispatcher=dispatcher,
        system_prompt="You are a test agent.",
        event_sink=event_sink,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        model="claude-opus-4-8",
    )


def tool_use_turn(
    tid: str = "tu_1",
    tool_name: str = "echo",
    tool_input: dict | None = None,
    text: str = "Let me check.",
    usage: UsageData | None = None,
) -> LLMTurnResult:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append(
        {"type": "tool_use", "id": tid, "name": tool_name, "input": tool_input or {"text": "hi"}}
    )
    return LLMTurnResult(
        content=content, stop_reason="tool_use", usage=usage or UsageData(100, 20, 90)
    )


def end_turn(text: str = "All done.", usage: UsageData | None = None) -> LLMTurnResult:
    return LLMTurnResult(
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=usage or UsageData(150, 10, 140),
    )


def text_delta(text: str) -> StreamDelta:
    return StreamDelta(kind="text", text=text)


async def test_two_turn_tool_use_flow(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="claude-opus-4-8", effort="high")
    llm = ScriptedLLM(
        turns=[tool_use_turn(), end_turn()],
        deltas=[[text_delta("Let me check.")], [text_delta("All done.")]],
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink)
    outcome = await loop.run(session.id, "hello world")

    assert outcome.ok is True
    assert outcome.stop_reason == "end_turn"
    assert outcome.iterations == 2
    assert outcome.usage == UsageData(input_tokens=250, output_tokens=30, cache_read=230)

    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0].content == [{"type": "text", "text": "hello world"}]

    assert messages[1].stop_reason == "tool_use"
    assert [b["type"] for b in messages[1].content] == ["text", "tool_use"]
    assert messages[1].input_tokens == 100 and messages[1].output_tokens == 20

    tool_result_block = messages[2].content[0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "tu_1"
    assert tool_result_block["is_error"] is False
    assert "echoed" in tool_result_block["content"]

    assert messages[3].stop_reason == "end_turn"
    assert messages[3].content[0]["text"] == "All done."

    assert needs_repair([{"role": m.role, "content": m.content} for m in messages]) is False

    calls = await repo.list_tool_calls(session.id)
    assert len(calls) == 1
    call = calls[0]
    assert call.tool_use_id == "tu_1"
    assert call.tool_name == "echo"
    assert call.message_id == messages[1].id
    assert call.is_error is False
    assert call.status == "ok"
    assert call.approval == "auto"
    assert "echoed" in str(call.result)

    assert [e.type for e in events] == [
        SSEEventType.TEXT_DELTA,
        SSEEventType.TOOL_CALL_START,
        SSEEventType.TOOL_CALL_RESULT,
        SSEEventType.TEXT_DELTA,
        SSEEventType.TURN_COMPLETE,
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    start = events[1].data
    assert start["tool_use_id"] == "tu_1" and start["name"] == "echo"
    assert "text" in start["input_preview"]
    result_ev = events[2].data
    assert result_ev["is_error"] is False and "echoed" in result_ev["output_preview"]
    complete = events[4].data
    assert complete["stop_reason"] == "end_turn"
    assert complete["usage"] == {
        "input_tokens": 250,
        "output_tokens": 30,
        "cache_read": 230,
        "cache_creation": 0,
    }

    assert len(llm.calls[0]["messages"]) == 1
    assert llm.calls[0]["messages"][0]["role"] == "user"
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[-1]["content"][0]["type"] == "tool_result"

    refreshed = await repo.get_session(session.id)
    assert refreshed.status == "idle"
    assert refreshed.total_input_tokens == 250
    assert refreshed.total_output_tokens == 30


async def test_event_seq_continues_across_runs(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(
        turns=[end_turn(), end_turn()],
        deltas=[[text_delta("one")], [text_delta("two")]],
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink)
    await loop.run(session.id, "first")
    await loop.run(session.id, "second")
    assert [e.seq for e in events] == [1, 2, 3, 4]


async def test_session_not_found_raises(repo: Repository, registry: ToolRegistry) -> None:
    loop = build_loop(repo, registry, ScriptedLLM([]))
    with pytest.raises(SessionNotFoundError):
        await loop.run("no-such-session", "hi")


async def test_refusal_is_terminal_no_retry(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(
        turns=[
            LLMTurnResult(
                content=[{"type": "text", "text": "I can't help with that."}],
                stop_reason="refusal",
                usage=UsageData(10, 5),
            )
        ]
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink)
    outcome = await loop.run(session.id, "do something bad")

    assert outcome.stop_reason == "refusal"
    assert len(llm.calls) == 1
    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].stop_reason == "refusal"
    assert events[-1].type == SSEEventType.TURN_COMPLETE
    assert events[-1].data["stop_reason"] == "refusal"
    refreshed = await repo.get_session(session.id)
    assert refreshed.status == "idle"


async def test_max_iterations_terminates_loop(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(
        turns=[
            tool_use_turn(tid="tu_1", usage=UsageData(10, 5)),
            tool_use_turn(tid="tu_2", usage=UsageData(10, 5)),
            tool_use_turn(tid="tu_3", usage=UsageData(10, 5)),
        ]
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink, max_iterations=3)
    outcome = await loop.run(session.id, "go")

    assert outcome.ok is False
    assert outcome.stop_reason == "max_iterations"
    assert outcome.iterations == 3
    assert len(llm.calls) == 3

    messages = await repo.get_messages(session.id)
    assert len(messages) == 7
    assert needs_repair([{"role": m.role, "content": m.content} for m in messages]) is False

    errors = [e for e in events if e.type == SSEEventType.ERROR]
    assert len(errors) == 1 and errors[0].data["code"] == "max_iterations"
    refreshed = await repo.get_session(session.id)
    assert refreshed.status == "idle"


async def test_max_tokens_rolls_back_snapshot_and_retries(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    truncated = LLMTurnResult(
        content=[
            {"type": "text", "text": "half wri"},
            {"type": "tool_use", "id": "tu_cut", "name": "echo", "input": {"text": "cu"}},
        ],
        stop_reason="max_tokens",
        usage=UsageData(10, 100),
    )
    llm = ScriptedLLM(turns=[truncated, end_turn(text="complete answer", usage=UsageData(12, 50))])
    loop = build_loop(repo, registry, llm, max_tokens=1000)
    outcome = await loop.run(session.id, "write something")

    assert outcome.ok is True
    assert outcome.stop_reason == "end_turn"
    assert llm.calls[0]["max_tokens"] == 1000
    assert llm.calls[1]["max_tokens"] == 2000
    assert len(llm.calls[1]["messages"]) == 1

    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content[0]["text"] == "complete answer"
    assert needs_repair([{"role": m.role, "content": m.content} for m in messages]) is False
    refreshed = await repo.get_session(session.id)
    assert refreshed.total_input_tokens == 22
    assert refreshed.total_output_tokens == 150


async def test_pause_turn_continues_conversation(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    paused = LLMTurnResult(
        content=[{"type": "text", "text": "long search part 1"}],
        stop_reason="pause_turn",
        usage=UsageData(50, 10),
    )
    llm = ScriptedLLM(turns=[paused, end_turn(usage=UsageData(60, 10))])
    loop = build_loop(repo, registry, llm)
    outcome = await loop.run(session.id, "search slowly")

    assert outcome.ok is True and outcome.stop_reason == "end_turn"
    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant", "assistant"]
    assert messages[1].stop_reason == "pause_turn"
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    assert second[1]["content"][0]["text"] == "long search part 1"


def test_merge_consecutive_same_role_messages() -> None:
    history = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    merged = _merge_consecutive(history)
    assert [m["role"] for m in merged] == ["user", "assistant"]
    assert len(merged[0]["content"]) == 2
    assert len(merged[1]["content"]) == 2
    assert len(history) == 4


async def test_fatal_llm_error_terminates_without_retry(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(turns=[LLMFatalError("invalid x-api-key", code="auth_error")])
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink)
    outcome = await loop.run(session.id, "hi")

    assert outcome.ok is False
    assert outcome.stop_reason == "error"
    assert outcome.error_code == "auth_error"
    assert len(llm.calls) == 1
    error_events = [e for e in events if e.type == SSEEventType.ERROR]
    assert len(error_events) == 1
    assert error_events[0].data == {"code": "auth_error", "message": "invalid x-api-key"}
    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user"]
    refreshed = await repo.get_session(session.id)
    assert refreshed.status == "idle"


async def test_generic_llm_error_normalized(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(turns=[RuntimeError("connection reset")])
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    loop = build_loop(repo, registry, llm, event_sink=sink)
    outcome = await loop.run(session.id, "hi")
    assert outcome.ok is False and outcome.error_code == "llm_error"
    assert any(e.type == SSEEventType.ERROR and e.data["code"] == "llm_error" for e in events)


class CancelledDispatcher:

    def __init__(self) -> None:
        self.notifier = None

    def set_approval_notifier(self, notifier: Any) -> None:
        self.notifier = notifier

    async def dispatch_many(self, calls: Any) -> list[Any]:
        raise asyncio.CancelledError()


async def test_interrupted_tool_execution_synthesizes_interrupted_results(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(turns=[tool_use_turn()])

    loop = AgentLoop(
        llm=llm,
        repo=repo,
        registry=registry,
        dispatcher=CancelledDispatcher(),
        system_prompt="test",
    )
    with pytest.raises(asyncio.CancelledError):
        await loop.run(session.id, "go")

    messages = await repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    block = messages[2].content[0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_1"
    assert block["is_error"] is True
    assert block["content"] == "interrupted: process crashed before this tool executed"
    assert needs_repair([{"role": m.role, "content": m.content} for m in messages]) is False

    calls = await repo.list_tool_calls(session.id)
    assert len(calls) == 1
    assert calls[0].status == "interrupted"
    assert calls[0].is_error is True

    refreshed = await repo.get_session(session.id)
    assert refreshed.status == "idle"


async def test_approval_flow_emits_approval_request(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(
        turns=[
            tool_use_turn(tid="tu_r", tool_name="risky", tool_input={"text": "rm"}),
            end_turn(),
        ]
    )
    events: list[SSEEvent] = []

    async def sink(event: SSEEvent) -> None:
        events.append(event)

    approve = AsyncMock(return_value=True)
    loop = build_loop(repo, registry, llm, event_sink=sink, approve_callback=approve)
    outcome = await loop.run(session.id, "do the risky thing")

    assert outcome.ok is True
    approve.assert_awaited_once_with("risky", {"text": "rm"})
    types = [e.type for e in events]
    assert types == [
        SSEEventType.TOOL_CALL_START,
        SSEEventType.APPROVAL_REQUEST,
        SSEEventType.TOOL_CALL_RESULT,
        SSEEventType.TURN_COMPLETE,
    ]
    approval_ev = events[1].data
    assert approval_ev["tool_use_id"] == "tu_r"
    assert approval_ev["name"] == "risky"
    assert approval_ev["input"] == {"text": "rm"}

    calls = await repo.list_tool_calls(session.id)
    assert calls[0].approval == "approved"


async def test_usage_and_ledger_consistency(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM(
        turns=[
            tool_use_turn(usage=UsageData(30, 8, 25)),
            end_turn(usage=UsageData(45, 6, 40)),
        ]
    )
    loop = build_loop(repo, registry, llm)
    outcome = await loop.run(session.id, "count tokens")

    refreshed = await repo.get_session(session.id)
    assert refreshed.total_input_tokens == outcome.usage.input_tokens == 75
    assert refreshed.total_output_tokens == outcome.usage.output_tokens == 14
    assert refreshed.cache_read_tokens == outcome.usage.cache_read == 65
    assert TokenLedger().should_compact() is False
