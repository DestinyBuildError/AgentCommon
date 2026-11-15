
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.database import build_engine
from app.db.models import Base, Message, ToolCall
from app.db.repository import Repository


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
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


async def test_create_and_get_session(repo: Repository) -> None:
    created = await repo.create_session(
        model="claude-opus-4-8", effort="high", title="demo", user_id="u1"
    )
    assert created.id and len(created.id) == 36
    assert created.status == "idle"
    assert created.total_input_tokens == 0
    assert created.total_output_tokens == 0
    assert created.cache_read_tokens == 0
    assert created.total_cost_usd == 0.0

    fetched = await repo.get_session(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "demo"
    assert fetched.user_id == "u1"

    assert await repo.get_session("nonexistent") is None


async def test_create_session_defaults(repo: Repository) -> None:
    s = await repo.create_session(model="claude-opus-4-8", effort="high")
    assert s.title is None
    assert s.user_id is None
    assert s.created_at is not None
    assert s.updated_at is not None


async def test_list_sessions_order_and_pagination(repo: Repository) -> None:
    s1 = await repo.create_session(model="m", effort="high", title="first")
    s2 = await repo.create_session(model="m", effort="high", title="second")

    sessions = await repo.list_sessions()
    assert {s.id for s in sessions} == {s1.id, s2.id}
    assert sessions[0].updated_at >= sessions[-1].updated_at

    page = await repo.list_sessions(limit=1)
    assert len(page) == 1

    await repo.update_session_status(s1.id, "running")
    sessions = await repo.list_sessions()
    assert sessions[0].id == s1.id


async def test_delete_session_cascades(repo: Repository, db_session: AsyncSession) -> None:
    s = await repo.create_session(model="m", effort="high")
    msg = await repo.append_message(s.id, "user", [{"type": "text", "text": "hi"}])
    await repo.record_tool_call(
        session_id=s.id,
        message_id=msg.id,
        tool_use_id="tu_1",
        tool_name="bash",
        tool_input={"command": "ls"},
        result={"stdout": ""},
        is_error=False,
        status="ok",
        approval="auto",
        duration_ms=5,
    )

    assert await repo.delete_session(s.id) is True
    assert await repo.get_session(s.id) is None

    orphan_msgs = (
        (await db_session.execute(select(Message).where(Message.session_id == s.id)))
        .scalars()
        .all()
    )
    orphan_calls = (
        (await db_session.execute(select(ToolCall).where(ToolCall.session_id == s.id)))
        .scalars()
        .all()
    )
    assert orphan_msgs == []
    assert orphan_calls == []

    assert await repo.delete_session(s.id) is False


async def test_update_session_status(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    running = await repo.update_session_status(s.id, "running")
    assert running is not None and running.status == "running"
    terminated = await repo.update_session_status(s.id, "terminated")
    assert terminated is not None and terminated.status == "terminated"
    assert await repo.update_session_status("nonexistent", "idle") is None


async def test_append_message_seq_monotonic_within_session(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    seqs = []
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        msg = await repo.append_message(s.id, role, [{"type": "text", "text": f"m{i}"}])
        seqs.append(msg.seq)
    assert seqs == [1, 2, 3, 4, 5]


async def test_seq_counter_is_per_session(repo: Repository) -> None:
    s1 = await repo.create_session(model="m", effort="high")
    s2 = await repo.create_session(model="m", effort="high")
    await repo.append_message(s1.id, "user", [{"type": "text", "text": "a"}])
    await repo.append_message(s1.id, "assistant", [{"type": "text", "text": "b"}])
    msg = await repo.append_message(s2.id, "user", [{"type": "text", "text": "c"}])
    assert msg.seq == 1


async def test_duplicate_session_seq_raises_integrity_error(
    repo: Repository, db_session: AsyncSession
) -> None:
    s = await repo.create_session(model="m", effort="high")
    first = await repo.append_message(s.id, "user", [{"type": "text", "text": "a"}])
    assert first.seq == 1

    duplicate = Message(
        id="dup-0000-0000-0000-000000000001",
        session_id=s.id,
        seq=first.seq,
        role="user",
        content=[{"type": "text", "text": "dup"}],
    )
    db_session.add(duplicate)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_get_messages_after_seq_cursor(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    for i in range(4):
        await repo.append_message(s.id, "user", [{"type": "text", "text": str(i)}])

    all_msgs = await repo.get_messages(s.id)
    assert [m.seq for m in all_msgs] == [1, 2, 3, 4]

    tail = await repo.get_messages(s.id, after_seq=2)
    assert [m.seq for m in tail] == [3, 4]

    empty = await repo.get_messages(s.id, after_seq=4)
    assert empty == []


async def test_append_message_records_usage_and_stop_reason(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    msg = await repo.append_message(
        s.id,
        "assistant",
        [{"type": "text", "text": "done"}],
        stop_reason="end_turn",
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    )
    assert msg.stop_reason == "end_turn"
    assert msg.input_tokens == 11
    assert msg.output_tokens == 7
    assert msg.cache_read == 3
    assert msg.cache_creation == 2

    msg2 = await repo.append_message(
        s.id,
        "assistant",
        [{"type": "text", "text": "x"}],
        usage={"input_tokens": 1, "output_tokens": 1, "cache_read": 9, "cache_creation": 8},
    )
    assert msg2.cache_read == 9
    assert msg2.cache_creation == 8

    msg3 = await repo.append_message(s.id, "user", [{"type": "text", "text": "y"}])
    assert (msg3.input_tokens, msg3.output_tokens, msg3.cache_read, msg3.cache_creation) == (0, 0, 0, 0)


async def test_update_session_usage_accumulates(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    await repo.update_session_usage(
        s.id, input_tokens=100, output_tokens=50, cache_read=10, cost_usd=0.01
    )
    await repo.update_session_usage(
        s.id, input_tokens=30, output_tokens=20, cache_read=5, cost_usd=0.02
    )
    updated = await repo.get_session(s.id)
    assert updated is not None
    assert updated.total_input_tokens == 130
    assert updated.total_output_tokens == 70
    assert updated.cache_read_tokens == 15
    assert abs(updated.total_cost_usd - 0.03) < 1e-9


async def test_update_session_usage_missing_session(repo: Repository) -> None:
    assert await repo.update_session_usage("nonexistent", input_tokens=1) is None


async def test_update_session_usage_zero_delta_touches_timestamp(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    before = s.updated_at
    after = await repo.update_session_usage(s.id)
    assert after is not None
    assert after.updated_at >= before


async def test_record_and_list_tool_calls(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    msg = await repo.append_message(
        s.id,
        "assistant",
        [{"type": "tool_use", "id": "tu_1", "name": "bash", "input": {}}],
    )

    ok_call = await repo.record_tool_call(
        session_id=s.id,
        message_id=msg.id,
        tool_use_id="tu_1",
        tool_name="bash",
        tool_input={"command": "ls"},
        result={"stdout": "file.txt"},
        is_error=False,
        status="ok",
        approval="auto",
        duration_ms=12,
    )
    assert ok_call.id and ok_call.duration_ms == 12
    assert ok_call.is_error is False
    assert ok_call.status == "ok"
    assert ok_call.tool_input == {"command": "ls"}
    assert ok_call.result == {"stdout": "file.txt"}

    err_call = await repo.record_tool_call(
        session_id=s.id,
        message_id=msg.id,
        tool_use_id="tu_2",
        tool_name="web_search",
        tool_input={"query": "x"},
        result={"error": {"type": "not_found", "message": "boom"}},
        is_error=True,
        status="error",
        approval="denied",
        duration_ms=30,
    )
    assert err_call.is_error is True
    assert err_call.approval == "denied"

    calls = await repo.list_tool_calls(s.id)
    assert len(calls) == 2
    assert {c.tool_name for c in calls} == {"bash", "web_search"}

    only_bash = await repo.list_tool_calls(s.id, tool_name="bash")
    assert [c.tool_use_id for c in only_bash] == ["tu_1"]

    assert await repo.list_tool_calls(s.id, tool_name="nonexistent_tool") == []
    assert await repo.list_tool_calls("nonexistent-session") == []


async def test_record_tool_call_defaults(repo: Repository) -> None:
    s = await repo.create_session(model="m", effort="high")
    msg = await repo.append_message(s.id, "assistant", [{"type": "text", "text": "t"}])
    call = await repo.record_tool_call(
        session_id=s.id,
        message_id=msg.id,
        tool_use_id="tu_x",
        tool_name="grep",
        tool_input={"pattern": "foo"},
    )
    assert call.is_error is False
    assert call.status == "ok"
    assert call.approval == "auto"
    assert call.duration_ms is None
    assert call.result is None
