
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.event_bus import SessionEventBus, derive_events_from_messages
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import (
    ErrorData,
    SSEEventType,
    TextDeltaData,
    make_event,
)


@pytest.fixture
async def db_factory():
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def loader(db_factory):
    async def load(session_id: str):
        async with db_factory() as session:
            return await Repository(session).get_messages(session_id)

    return load


async def take(agen: AsyncIterator, n: int, timeout: float = 1.0) -> list:
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(anext(agen), timeout))
    return out


async def test_live_subscription_in_order():
    bus = SessionEventBus()
    agen = bus.subscribe("s1", after_seq=0)
    task = asyncio.create_task(take(agen, 2))
    await asyncio.sleep(0)
    bus.publish("s1", make_event(1, SSEEventType.TEXT_DELTA, TextDeltaData(text="a")))
    bus.publish("s1", make_event(2, SSEEventType.ERROR, ErrorData(code="x", message="y")))
    events = await task
    assert [e.seq for e in events] == [1, 2]
    assert [e.type for e in events] == [SSEEventType.TEXT_DELTA, SSEEventType.ERROR]
    await agen.aclose()


async def test_buffered_replay_after_cursor():
    bus = SessionEventBus()
    for seq in range(1, 4):
        bus.publish("s1", make_event(seq, SSEEventType.TEXT_DELTA, TextDeltaData(text=str(seq))))
    agen = bus.subscribe("s1", after_seq=1)
    replayed = await take(agen, 2)
    assert [e.seq for e in replayed] == [2, 3]
    bus.publish("s1", make_event(4, SSEEventType.TEXT_DELTA, TextDeltaData(text="4")))
    live = await take(agen, 1)
    assert live[0].seq == 4
    await agen.aclose()


async def test_caught_up_cursor_replays_nothing():
    bus = SessionEventBus()
    bus.publish("s1", make_event(1, SSEEventType.TEXT_DELTA, TextDeltaData(text="a")))
    agen = bus.subscribe("s1", after_seq=1)
    bus.publish("s1", make_event(2, SSEEventType.TEXT_DELTA, TextDeltaData(text="b")))
    events = await take(agen, 1)
    assert [e.seq for e in events] == [2]
    await agen.aclose()


async def test_two_subscribers_isolated_cursors():
    bus = SessionEventBus()
    for seq in range(1, 4):
        bus.publish("s1", make_event(seq, SSEEventType.TEXT_DELTA, TextDeltaData(text=str(seq))))
    a = bus.subscribe("s1", after_seq=0)
    b = bus.subscribe("s1", after_seq=2)
    assert [e.seq for e in await take(a, 3)] == [1, 2, 3]
    assert [e.seq for e in await take(b, 1)] == [3]
    bus.publish("s1", make_event(4, SSEEventType.TEXT_DELTA, TextDeltaData(text="4")))
    assert (await take(a, 1))[0].seq == 4
    assert (await take(b, 1))[0].seq == 4
    await a.aclose()
    await b.aclose()


async def test_buffer_overflow_drops_oldest():
    bus = SessionEventBus(capacity=3)
    for seq in range(1, 6):
        bus.publish("s1", make_event(seq, SSEEventType.TEXT_DELTA, TextDeltaData(text=str(seq))))
    agen = bus.subscribe("s1", after_seq=0)
    replayed = await take(agen, 3)
    assert [e.seq for e in replayed] == [3, 4, 5]
    assert bus.last_seq("s1") == 5
    await agen.aclose()


async def test_close_session_ends_subscriber():
    bus = SessionEventBus()
    agen = bus.subscribe("s1", after_seq=0)
    bus.publish("s1", make_event(1, SSEEventType.TEXT_DELTA, TextDeltaData(text="a")))
    await take(agen, 1)
    bus.close_session("s1")
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(agen), 1.0)


async def test_emit_assigns_consecutive_seq():
    bus = SessionEventBus()
    ev1 = bus.emit("s1", SSEEventType.TEXT_DELTA, TextDeltaData(text="a"))
    ev2 = bus.emit("s1", SSEEventType.TEXT_DELTA, TextDeltaData(text="b"))
    assert (ev1.seq, ev2.seq) == (1, 2)
    assert bus.last_seq("s1") == 2
    ev3 = bus.emit("s1", SSEEventType.TEXT_DELTA, TextDeltaData(text="c"))
    assert ev3.seq == 3


async def seed_tool_flow(db_factory, session_id: str) -> None:
    async with db_factory() as db:
        repo = Repository(db)
        await repo.create_session(model="m", effort="high")
        await repo.append_message(session_id, "user", [{"type": "text", "text": "你好"}])
        await repo.append_message(
            session_id,
            "assistant",
            [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"text": "hi"}},
            ],
            stop_reason="tool_use",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        await repo.append_message(
            session_id,
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": '{"echoed": "hi"}',
                    "is_error": False,
                }
            ],
        )
        await repo.append_message(
            session_id,
            "assistant",
            [{"type": "text", "text": "All done."}],
            stop_reason="end_turn",
            usage={"input_tokens": 20, "output_tokens": 6},
        )


async def test_db_derivation_backfills_history(loader, db_factory):
    session_id = "sess-1"
    await seed_tool_flow(db_factory, session_id)
    bus = SessionEventBus(loader)
    agen = bus.subscribe(session_id, after_seq=0)
    events = await take(agen, 6)
    assert [e.type for e in events] == [
        SSEEventType.TEXT_DELTA,
        SSEEventType.TEXT_DELTA,
        SSEEventType.TOOL_CALL_START,
        SSEEventType.TOOL_CALL_RESULT,
        SSEEventType.TEXT_DELTA,
        SSEEventType.TURN_COMPLETE,
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5, 6]
    start = events[2].data
    assert start["tool_use_id"] == "tu_1" and start["name"] == "echo"
    result = events[3].data
    assert result["is_error"] is False and "echoed" in result["output_preview"]
    complete = events[5].data
    assert complete["stop_reason"] == "end_turn"
    assert complete["usage"]["input_tokens"] == 30
    assert complete["usage"]["output_tokens"] == 11
    bus.publish(session_id, make_event(7, SSEEventType.TEXT_DELTA, TextDeltaData(text="live")))
    live = await take(agen, 1)
    assert live[0].seq == 7
    await agen.aclose()


async def test_db_derivation_ignores_stale_cursor_full_resync(loader, db_factory):
    session_id = "sess-2"
    await seed_tool_flow(db_factory, session_id)
    bus = SessionEventBus(loader)
    agen = bus.subscribe(session_id, after_seq=999)
    events = await take(agen, 6)
    assert [e.seq for e in events] == [1, 2, 3, 4, 5, 6]
    await agen.aclose()


async def test_second_subscriber_replays_from_hydrated_buffer(loader, db_factory):
    session_id = "sess-3"
    await seed_tool_flow(db_factory, session_id)
    bus = SessionEventBus(loader)
    first = bus.subscribe(session_id, after_seq=0)
    await take(first, 6)
    await first.aclose()
    second = bus.subscribe(session_id, after_seq=0)
    assert [e.seq for e in await take(second, 6)] == [1, 2, 3, 4, 5, 6]
    await second.aclose()


async def test_derivation_with_empty_db_yields_nothing(loader):
    bus = SessionEventBus(loader)
    agen = bus.subscribe("no-such-session", after_seq=0)
    bus.publish("no-such-session", make_event(1, SSEEventType.TEXT_DELTA, TextDeltaData(text="x")))
    events = await take(agen, 1)
    assert events[0].seq == 1
    await agen.aclose()


async def test_derive_events_from_messages_pure_function():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}],
         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "t9", "name": "calc", "input": {"x": 1}},
        ], "stop_reason": "tool_use",
         "input_tokens": 5, "output_tokens": 2, "cache_read": 1, "cache_creation": 0},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t9", "content": "boom", "is_error": True},
        ], "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
    ]
    events = derive_events_from_messages(messages)
    assert [e.type for e in events] == [
        SSEEventType.TEXT_DELTA,
        SSEEventType.THINKING_DELTA,
        SSEEventType.TOOL_CALL_START,
        SSEEventType.TOOL_CALL_RESULT,
    ]
    assert events[1].data["text"] == "hmm"
    assert events[3].data["is_error"] is True and events[3].data["output_preview"] == "boom"
    assert all(e.type is not SSEEventType.TURN_COMPLETE for e in events)

    with_terminal = derive_events_from_messages([
        {"role": "user", "content": [{"type": "text", "text": "q"}],
         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}],
         "stop_reason": "end_turn",
         "input_tokens": 7, "output_tokens": 3, "cache_read": 2, "cache_creation": 0},
    ])
    assert with_terminal[-1].type is SSEEventType.TURN_COMPLETE
    assert with_terminal[-1].data["stop_reason"] == "end_turn"
    assert with_terminal[-1].data["usage"]["input_tokens"] == 7

    user_only = derive_events_from_messages([
        {"role": "user", "content": [{"type": "text", "text": "q"}],
         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
    ])
    assert [e.type for e in user_only] == [SSEEventType.TEXT_DELTA]

    assert derive_events_from_messages([]) == []


def test_derive_events_carry_user_role():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "帮我查一下"}],
         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
        {"role": "assistant", "content": [{"type": "text", "text": "好的"}],
         "stop_reason": "end_turn",
         "input_tokens": 1, "output_tokens": 1, "cache_read": 0, "cache_creation": 0},
    ]
    events = derive_events_from_messages(messages)
    user_event, assistant_event = events[0], events[1]
    assert user_event.type is SSEEventType.TEXT_DELTA
    assert user_event.data["role"] == "user"
    assert user_event.data["text"] == "帮我查一下"
    assert assistant_event.type is SSEEventType.TEXT_DELTA
    assert "role" not in assistant_event.data


async def test_strict_cursor_with_derived_history_and_live_continuation(loader, db_factory):
    session_id = "sess-strict"
    await seed_tool_flow(db_factory, session_id)
    bus = SessionEventBus(loader)
    warmup = bus.subscribe(session_id, after_seq=0)
    await take(warmup, 6)
    await warmup.aclose()
    bus.publish(session_id, make_event(7, SSEEventType.TEXT_DELTA, TextDeltaData(text="live1")))
    bus.publish(session_id, make_event(8, SSEEventType.TEXT_DELTA, TextDeltaData(text="live2")))

    agen = bus.subscribe(session_id, after_seq=2)
    assert [e.seq for e in await take(agen, 6)] == [3, 4, 5, 6, 7, 8]
    await agen.aclose()

    agen = bus.subscribe(session_id, after_seq=7)
    assert [e.seq for e in await take(agen, 1)] == [8]
    await agen.aclose()

    agen = bus.subscribe(session_id, after_seq=999)
    assert [e.seq for e in await take(agen, 8)] == [1, 2, 3, 4, 5, 6, 7, 8]
    await agen.aclose()
