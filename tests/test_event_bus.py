"""SessionEventBus 测试 —— 缓冲重放/去重、实时接续、DB 推导补历史、
溢出丢最老、订阅关闭、emit 自动编号。"""

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


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------


@pytest.fixture
async def db_factory():
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # 内存库共享单连接
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


# ----------------------------------------------------------------------
# 实时流 / 缓冲重放
# ----------------------------------------------------------------------


async def test_live_subscription_in_order():
    bus = SessionEventBus()
    agen = bus.subscribe("s1", after_seq=0)
    task = asyncio.create_task(take(agen, 2))
    await asyncio.sleep(0)  # 让订阅者先注册
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
    assert [e.seq for e in replayed] == [2, 3]  # 只重放 seq > 1
    # 实时接续
    bus.publish("s1", make_event(4, SSEEventType.TEXT_DELTA, TextDeltaData(text="4")))
    live = await take(agen, 1)
    assert live[0].seq == 4
    await agen.aclose()


async def test_caught_up_cursor_replays_nothing():
    bus = SessionEventBus()
    bus.publish("s1", make_event(1, SSEEventType.TEXT_DELTA, TextDeltaData(text="a")))
    agen = bus.subscribe("s1", after_seq=1)  # 已追平
    bus.publish("s1", make_event(2, SSEEventType.TEXT_DELTA, TextDeltaData(text="b")))
    events = await take(agen, 1)
    assert [e.seq for e in events] == [2]  # 只有实时事件，不重放
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
    bus = SessionEventBus(capacity=3)  # 无 loader：DB 推导不可用
    for seq in range(1, 6):
        bus.publish("s1", make_event(seq, SSEEventType.TEXT_DELTA, TextDeltaData(text=str(seq))))
    agen = bus.subscribe("s1", after_seq=0)
    replayed = await take(agen, 3)
    assert [e.seq for e in replayed] == [3, 4, 5]  # 最老的 1、2 已被挤出
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
    # 与显式 publish 的 seq 空间一致（单调）
    ev3 = bus.emit("s1", SSEEventType.TEXT_DELTA, TextDeltaData(text="c"))
    assert ev3.seq == 3


# ----------------------------------------------------------------------
# DB 推导补历史（重启 / 冷会话场景）
# ----------------------------------------------------------------------


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
    bus = SessionEventBus(loader)  # 全新总线：无任何缓冲（模拟重启后）
    agen = bus.subscribe(session_id, after_seq=0)
    events = await take(agen, 6)
    assert [e.type for e in events] == [
        SSEEventType.TEXT_DELTA,        # user 你好
        SSEEventType.TEXT_DELTA,        # assistant Let me check.
        SSEEventType.TOOL_CALL_START,   # tu_1
        SSEEventType.TOOL_CALL_RESULT,  # tu_1
        SSEEventType.TEXT_DELTA,        # assistant All done.
        SSEEventType.TURN_COMPLETE,
    ]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5, 6]  # 推导事件重新连续编号
    start = events[2].data
    assert start["tool_use_id"] == "tu_1" and start["name"] == "echo"
    result = events[3].data
    assert result["is_error"] is False and "echoed" in result["output_preview"]
    complete = events[5].data
    assert complete["stop_reason"] == "end_turn"
    assert complete["usage"]["input_tokens"] == 30  # 10 + 20
    assert complete["usage"]["output_tokens"] == 11  # 5 + 6
    # 实时事件接在推导流之后：seq 从 7 连续
    bus.publish(session_id, make_event(7, SSEEventType.TEXT_DELTA, TextDeltaData(text="live")))
    live = await take(agen, 1)
    assert live[0].seq == 7
    await agen.aclose()


async def test_db_derivation_ignores_stale_cursor_full_resync(loader, db_factory):
    """重启后客户端带着旧 seq 空间的大游标重连 → 全量重同步（推导事件重放）。"""
    session_id = "sess-2"
    await seed_tool_flow(db_factory, session_id)
    bus = SessionEventBus(loader)
    agen = bus.subscribe(session_id, after_seq=999)
    events = await take(agen, 6)
    assert [e.seq for e in events] == [1, 2, 3, 4, 5, 6]  # 游标失效也补齐历史
    await agen.aclose()


async def test_second_subscriber_replays_from_hydrated_buffer(loader, db_factory):
    """第一个订阅者触发推导后，后续订阅者直接从缓冲重放（不再打 DB）。"""
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
    assert events[0].seq == 1  # 无历史可推导，直接进实时
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
    # 列表以 user（tool_result）结尾 = run 进行中/崩溃残留 → 不合成 turn_complete
    # （真实模式缺陷 #2：0 用量幻影收尾事件会诱导前端误判轮次结束）
    assert all(e.type is not SSEEventType.TURN_COMPLETE for e in events)

    # 列表以 assistant 结尾 → 合成 turn_complete（stop_reason/usage 如实累计）
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

    # 全部为 user 消息（run 尚未产出 assistant）→ 同样不合成收尾事件
    user_only = derive_events_from_messages([
        {"role": "user", "content": [{"type": "text", "text": "q"}],
         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0},
    ])
    assert [e.type for e in user_only] == [SSEEventType.TEXT_DELTA]

    # 空消息列表 → 不产出任何事件（回归 minor-1）
    assert derive_events_from_messages([]) == []


def test_derive_events_carry_user_role():
    """P3 联调遗留①：推导历史的 user text 带 role="user"——重启后刷新页面
    用户消息渲染为用户样式；assistant 文本不带 role（与实时流 wire 一致）。"""
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
    assert "role" not in assistant_event.data  # exclude_none：实时 wire 不加键


async def test_strict_cursor_with_derived_history_and_live_continuation(loader, db_factory):
    """推导历史 + 实时事件并存、游标在本进程 seq 空间内 → 严格只重放
    seq > after_seq（回归 minor-2：不得触发全量重放）。"""
    session_id = "sess-strict"
    await seed_tool_flow(db_factory, session_id)  # DB 推导 → 6 条事件
    bus = SessionEventBus(loader)
    # 先订阅一次触发推导（缓冲填充 1..6）
    warmup = bus.subscribe(session_id, after_seq=0)
    await take(warmup, 6)
    await warmup.aclose()
    # 追加实时事件 7、8（模拟 run 进行中）
    bus.publish(session_id, make_event(7, SSEEventType.TEXT_DELTA, TextDeltaData(text="live1")))
    bus.publish(session_id, make_event(8, SSEEventType.TEXT_DELTA, TextDeltaData(text="live2")))

    # 游标落在推导段内：严格补 3..8
    agen = bus.subscribe(session_id, after_seq=2)
    assert [e.seq for e in await take(agen, 6)] == [3, 4, 5, 6, 7, 8]
    await agen.aclose()

    # 游标落在实时段内（> 推导上界）：严格只补 8，绝不回吐推导段
    agen = bus.subscribe(session_id, after_seq=7)
    assert [e.seq for e in await take(agen, 1)] == [8]
    await agen.aclose()

    # 跨重启旧游标（> last_seq）→ 仍全量重同步
    agen = bus.subscribe(session_id, after_seq=999)
    assert [e.seq for e in await take(agen, 8)] == [1, 2, 3, 4, 5, 6, 7, 8]
    await agen.aclose()
