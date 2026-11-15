"""SessionManager 测试 —— 全部 mock LLM，绝不发起真实 API 调用。

覆盖：连发 3 条消息按序消费、同一会话不并发两个消费者、审批 Future 的
allow / deny / 超时 deny（0.05s 短超时注入）、interrupt（审批挂起点打断 +
LLM 挂起点打断）、空闲超时回收、启动恢复 repair_orphan_sessions、
跨 run 事件 seq 连续。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.dispatcher import ToolDispatcher
from app.core.event_bus import SessionEventBus
from app.core.llm_client import LLMTurnResult, StreamDelta, UsageData
from app.core.repair import SYNTHETIC_INTERRUPTED
from app.core.session_manager import SessionManager, repair_orphan_sessions
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import SSEEvent, SSEEventType
from app.tools.registry import ToolRegistry, ToolSpec


# ----------------------------------------------------------------------
# 夹具：文件库（避免 StaticPool 并发事务串扰）+ Registry + Scripted LLM
# ----------------------------------------------------------------------


@pytest.fixture
async def db_factory(tmp_path):
    engine_url = f"sqlite+aiosqlite:///{(tmp_path / 'sm_test.db').as_posix()}"
    from app.db.database import build_engine

    engine = build_engine(engine_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


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
            access="execute",  # 审批门由权限矩阵按 access 判定
        )
    )
    return registry


@pytest.fixture
def loader(db_factory):
    async def load(session_id: str):
        async with db_factory() as db:
            return await Repository(db).get_messages(session_id)

    return load


class ScriptedLLM:
    """脚本化 mock LLM：按序回放 turns（可附 deltas）；异常实例直接抛出。"""

    def __init__(
        self,
        turns: list[Any] | None = None,
        deltas: list[list[StreamDelta]] | None = None,
    ) -> None:
        self.turns: list[Any] = list(turns or [])
        self.deltas: list[list[StreamDelta]] = list(deltas or [])
        self.calls: list[dict[str, Any]] = []

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.calls.append({k: deepcopy(v) for k, v in kwargs.items() if k != "on_delta"})
        on_delta = kwargs.get("on_delta")
        if self.deltas:
            for delta in self.deltas.pop(0):
                if on_delta is not None:
                    await on_delta(delta)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn

    def add(self, turn: Any) -> None:
        self.turns.append(turn)


class ConcurrencyTrackingLLM:
    """包装 ScriptedLLM：统计 run_turn 最大并发数（验证单消费者）。"""

    def __init__(self, inner: ScriptedLLM) -> None:
        self.inner = inner
        self.active = 0
        self.max_active = 0

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return await self.inner.run_turn(**kwargs)
        finally:
            self.active -= 1


class BlockingLLM:
    """挂在 run_turn 上的 mock LLM（interrupt 测试用）。"""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return LLMTurnResult(
            content=[{"type": "text", "text": "unblocked"}],
            stop_reason="end_turn",
            usage=UsageData(10, 5),
        )


def end_turn(text: str = "done", usage: UsageData | None = None) -> LLMTurnResult:
    return LLMTurnResult(
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=usage or UsageData(150, 10, 140),
    )


def tool_use_turn(tid: str, tool_name: str, tool_input: dict) -> LLMTurnResult:
    return LLMTurnResult(
        content=[
            {"type": "text", "text": "working"},
            {"type": "tool_use", "id": tid, "name": tool_name, "input": tool_input},
        ],
        stop_reason="tool_use",
        usage=UsageData(100, 20, 90),
    )


def make_manager(
    db_factory,
    loader,
    registry: ToolRegistry,
    llm: Any,
    *,
    approval_timeout_s: float = 120.0,
    idle_timeout_s: float = 600.0,
) -> SessionManager:
    bus = SessionEventBus(loader)

    def dispatcher_factory(approve_callback):
        return ToolDispatcher(
            registry, approve_callback=approve_callback, retry_backoff_base_s=0
        )

    manager = SessionManager(
        session_factory=db_factory,
        bus=bus,
        llm=llm,
        registry=registry,
        dispatcher_factory=dispatcher_factory,
        system_prompt="test",
        approval_timeout_s=approval_timeout_s,
        idle_timeout_s=idle_timeout_s,
    )
    _ACTIVE_MANAGERS.append(manager)  # autouse 夹具兜底回收，防用例失败泄漏消费者
    return manager


_ACTIVE_MANAGERS: list[SessionManager] = []


@pytest.fixture(autouse=True)
async def _stop_leftover_consumers():
    """测试隔离兜底：无论用例断言成败，都停掉其创建的会话消费者任务，
    防止未收尾的任务泄漏到后续用例（跨用例顺序依赖的根源之一）。"""
    yield
    for manager in _ACTIVE_MANAGERS:
        try:
            await manager.stop_all()
        except Exception:
            pass
    _ACTIVE_MANAGERS.clear()


async def wait_until(pred, timeout: float = 3.0, interval: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


async def wait_run_done(manager: SessionManager, session_id: str, timeout: float = 3.0) -> None:
    """等待当前 run 结束（消费者仍在跑、只是空闲，不算结束）。"""
    await wait_until(
        lambda: (
            manager.get_running(session_id) is None
            or manager.get_running(session_id).current_run is None
        ),
        timeout=timeout,
    )


async def create_session(db_factory, model: str = "m") -> str:
    async with db_factory() as db:
        session = await Repository(db).create_session(model=model, effort="high")
        return session.id


async def get_messages(db_factory, session_id: str):
    async with db_factory() as db:
        return await Repository(db).get_messages(session_id)


async def get_session_row(db_factory, session_id: str):
    async with db_factory() as db:
        return await Repository(db).get_session(session_id)


# ----------------------------------------------------------------------
# 消息队列与单消费者
# ----------------------------------------------------------------------


async def test_three_messages_consumed_in_order(db_factory, loader, registry):
    llm = ScriptedLLM([end_turn("reply one"), end_turn("reply two"), end_turn("reply three")])
    tracker = ConcurrencyTrackingLLM(llm)
    manager = make_manager(db_factory, loader, registry, tracker)
    session_id = await create_session(db_factory)

    manager.enqueue_message(session_id, "msg-1")
    manager.enqueue_message(session_id, "msg-2")
    manager.enqueue_message(session_id, "msg-3")

    await wait_until(lambda: len(llm.calls) == 3)
    await wait_run_done(manager, session_id)

    messages = await get_messages(db_factory, session_id)
    texts = [m.content[0]["text"] for m in messages if m.role == "user"]
    assert texts == ["msg-1", "msg-2", "msg-3"]  # 严格按入队顺序消费
    replies = [
        m.content[0]["text"] for m in messages if m.role == "assistant"
    ]
    assert replies == ["reply one", "reply two", "reply three"]
    assert tracker.max_active == 1  # 同一会话绝不并发两个 run
    row = await get_session_row(db_factory, session_id)
    assert row.status == "idle"
    await manager.stop_all()


async def test_events_seq_continuous_across_runs(db_factory, loader, registry):
    def text_delta(t: str) -> StreamDelta:
        return StreamDelta(kind="text", text=t)

    llm = ScriptedLLM(
        [end_turn("a"), end_turn("b")],
        deltas=[[text_delta("a")], [text_delta("b")]],
    )
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)

    events: list[SSEEvent] = []
    agen = manager.bus.subscribe(session_id, after_seq=0)

    manager.enqueue_message(session_id, "first")
    manager.enqueue_message(session_id, "second")
    await wait_until(lambda: len(llm.calls) == 2)
    for _ in range(4):  # 每 run：text_delta + turn_complete
        events.append(await asyncio.wait_for(anext(agen), 2.0))
    await agen.aclose()

    seqs = [e.seq for e in events]
    assert seqs == [1, 2, 3, 4]  # 跨 run 接续：每 run 的 loop 从 bus.last_seq 续编
    assert [e.type for e in events] == [
        SSEEventType.TEXT_DELTA,
        SSEEventType.TURN_COMPLETE,
        SSEEventType.TEXT_DELTA,
        SSEEventType.TURN_COMPLETE,
    ]
    assert events[2].data["text"] == "b"
    await manager.stop_all()


async def test_start_session_idempotent(db_factory, loader, registry):
    llm = ScriptedLLM([])
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    rs1 = manager.start_session(session_id)
    rs2 = manager.start_session(session_id)
    assert rs1 is rs2  # 同一会话只有一个消费者任务
    await manager.stop_session(session_id)


# ----------------------------------------------------------------------
# 审批门：allow / deny / 超时 deny
# ----------------------------------------------------------------------


async def wait_for_approval(manager, session_id, tool_use_id, timeout=3.0):
    """等待一个「可决议」的 pending 审批：条目存在且 Future 尚未决议。

    只查成员资格不够——已决议（done）的旧条目在重注册前仍留在字典里，
    会造成对旧 Future 的迟到决议（返回 False）。
    """

    def _resolvable() -> bool:
        rs = manager.get_running(session_id)
        if rs is None:
            return False
        entry = rs.pending_approvals.get(tool_use_id)
        return entry is not None and not entry.future.done()

    await wait_until(_resolvable, timeout=timeout)


async def approval_flow(db_factory, loader, registry, allow: bool, timeout_s: float = 120.0):
    """公共流程：risky 工具 → 审批 → end_turn。"""
    llm = ScriptedLLM([tool_use_turn("tu_r", "risky", {"text": "rm"}), end_turn()])
    manager = make_manager(db_factory, loader, registry, llm, approval_timeout_s=timeout_s)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "do it")
    await wait_for_approval(manager, session_id, "tu_r")
    return llm, manager, session_id


async def test_approval_allow_continues_flow(db_factory, loader, registry):
    llm, manager, session_id = await approval_flow(db_factory, loader, registry, allow=True)
    resolved = await manager.resolve_approval(session_id, "tu_r", True)
    assert resolved is True
    await wait_until(lambda: len(llm.calls) == 2)  # 审批放行 → 第二轮 LLM 调用
    await wait_run_done(manager, session_id)

    messages = await get_messages(db_factory, session_id)
    tool_result = next(
        b for m in messages for b in m.content if b.get("type") == "tool_result"
    )
    assert tool_result["is_error"] is False

    async with db_factory() as db:
        calls = await Repository(db).list_tool_calls(session_id)
    assert calls[0].approval == "approved"
    assert calls[0].status == "ok"
    await manager.stop_all()


async def test_approval_deny_rejects_tool(db_factory, loader, registry):
    llm, manager, session_id = await approval_flow(db_factory, loader, registry, allow=False)
    resolved = await manager.resolve_approval(session_id, "tu_r", False)
    assert resolved is True
    await wait_until(lambda: len(llm.calls) == 2)
    await wait_run_done(manager, session_id)

    messages = await get_messages(db_factory, session_id)
    tool_result = next(
        b for m in messages for b in m.content if b.get("type") == "tool_result"
    )
    assert tool_result["is_error"] is True
    assert "user denied" in tool_result["content"]

    async with db_factory() as db:
        calls = await Repository(db).list_tool_calls(session_id)
    assert calls[0].approval == "denied"
    assert calls[0].status == "denied"
    await manager.stop_all()


async def test_approval_timeout_auto_denies(db_factory, loader, registry):
    llm, manager, session_id = await approval_flow(
        db_factory, loader, registry, allow=False, timeout_s=0.05
    )
    await asyncio.sleep(0.2)  # 等审批超时（无人 resolve）
    await wait_until(lambda: len(llm.calls) == 2)
    await wait_run_done(manager, session_id)

    messages = await get_messages(db_factory, session_id)
    tool_result = next(
        b for m in messages for b in m.content if b.get("type") == "tool_result"
    )
    assert tool_result["is_error"] is True  # 超时 → deny → is_error 链路
    # 超时后迟到决议 → False（API 层据此 404）
    assert await manager.resolve_approval(session_id, "tu_r", True) is False
    await manager.stop_all()


async def test_resolve_approval_unknown_pending_returns_false(db_factory, loader, registry):
    llm = ScriptedLLM([])
    manager = make_manager(db_factory, loader, registry, llm)
    assert await manager.resolve_approval("ghost-session", "tu_x", True) is False
    session_id = await create_session(db_factory)
    assert await manager.resolve_approval(session_id, "tu_x", True) is False


# ----------------------------------------------------------------------
# interrupt
# ----------------------------------------------------------------------


async def test_interrupt_during_approval_synthesizes_interrupted_results(
    db_factory, loader, registry
):
    llm = ScriptedLLM([tool_use_turn("tu_r", "risky", {"text": "rm"})])
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "go")
    await wait_for_approval(manager, session_id, "tu_r")

    assert await manager.interrupt(session_id) is True
    await wait_run_done(manager, session_id)

    # 引擎 finally 已合成 interrupted tool_result，不变量 1 成立
    messages = await get_messages(db_factory, session_id)
    tool_result = next(
        b for m in messages for b in m.content if b.get("type") == "tool_result"
    )
    assert tool_result["tool_use_id"] == "tu_r"
    assert tool_result["is_error"] is True
    assert tool_result["content"] == SYNTHETIC_INTERRUPTED
    row = await get_session_row(db_factory, session_id)
    assert row.status == "idle"

    # 中断后消费者仍存活：来一条新消息照常处理
    llm.add(end_turn("recovered"))
    manager.enqueue_message(session_id, "again")
    await wait_until(lambda: len(llm.calls) == 2)
    await wait_run_done(manager, session_id)  # 等 run 收尾落库再读历史（防消息读取竞态）
    messages = await get_messages(db_factory, session_id)
    assert messages[-1].content[0]["text"] == "recovered"
    await manager.stop_all()


async def test_interrupt_during_llm_call(db_factory, loader, registry):
    blocking = BlockingLLM()
    manager = make_manager(db_factory, loader, registry, blocking)
    session_id = await create_session(db_factory)

    events: list[SSEEvent] = []
    agen = manager.bus.subscribe(session_id, after_seq=0)
    manager.enqueue_message(session_id, "hang")
    await wait_until(lambda: blocking.started.is_set())

    assert await manager.interrupt(session_id) is True
    assert blocking.cancelled is True
    await wait_run_done(manager, session_id)

    # SessionManager 兜底补发 turn_complete(interrupted)（引擎在 LLM 阶段
    # 被打断时不发任何事件——见 agent_loop 的 CancelledError 路径）
    events.append(await asyncio.wait_for(anext(agen), 2.0))
    assert events[0].type is SSEEventType.TURN_COMPLETE
    assert events[0].data["stop_reason"] == "interrupted"
    assert events[0].seq == 1
    await agen.aclose()

    messages = await get_messages(db_factory, session_id)
    assert [m.role for m in messages] == ["user"]  # 没有半截 assistant 落库
    row = await get_session_row(db_factory, session_id)
    assert row.status == "idle"
    await manager.stop_all()


async def test_interrupt_without_active_run_returns_false(db_factory, loader, registry):
    llm = ScriptedLLM([])
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    assert await manager.interrupt(session_id) is False  # 会话未启动
    manager.start_session(session_id)
    assert await manager.interrupt(session_id) is False  # 无 run
    assert await manager.interrupt("ghost") is False
    await manager.stop_session(session_id)


# ----------------------------------------------------------------------
# 空闲超时 / 停止
# ----------------------------------------------------------------------


async def test_idle_timeout_stops_consumer(db_factory, loader, registry):
    llm = ScriptedLLM([end_turn()])
    manager = make_manager(db_factory, loader, registry, llm, idle_timeout_s=0.05)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "hi")
    await wait_until(lambda: len(llm.calls) == 1)
    await wait_run_done(manager, session_id)
    # 消费者已退出，会话可重新唤起
    llm.add(end_turn("revived"))
    manager.enqueue_message(session_id, "back")
    await wait_until(lambda: len(llm.calls) == 2)
    await wait_run_done(manager, session_id)  # 等 run 收尾落库再读历史（防消息读取竞态）
    messages = await get_messages(db_factory, session_id)
    assert messages[-1].content[0]["text"] == "revived"


async def test_stop_session_cancels_run_and_consumer(db_factory, loader, registry):
    blocking = BlockingLLM()
    manager = make_manager(db_factory, loader, registry, blocking)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "hang")
    await wait_until(lambda: blocking.started.is_set())
    assert await manager.stop_session(session_id) is True
    await wait_run_done(manager, session_id)
    assert blocking.cancelled is True
    row = await get_session_row(db_factory, session_id)
    assert row.status == "idle"  # 引擎 finally 归位
    assert await manager.stop_session(session_id) is False  # 已不在跑


async def test_external_cancel_terminates_consumer(db_factory, loader, registry):
    """事件循环关停（asyncio._cancel_all_tasks）只取消消费者任务本身时，
    消费者必须消亡——不得把外部取消误判为 interrupt 而继续存活
    （major：进程/测试收尾挂死的根因）。"""
    blocking = BlockingLLM()
    manager = make_manager(db_factory, loader, registry, blocking)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "hang")
    await wait_until(lambda: blocking.started.is_set())
    rs = manager.get_running(session_id)
    assert rs is not None and rs.current_run is not None
    rs.task.cancel()  # 模拟事件循环关停：只取消消费者任务本身
    done, pending = await asyncio.wait({rs.task}, timeout=3.0)
    assert rs.task in done and rs.task not in pending  # 消费者已消亡
    # 子 run 任务随后由 stop_all 收尾（引擎合成 interrupted 结果并落库）
    await manager.stop_all()
    row = await get_session_row(db_factory, session_id)
    assert row.status == "idle"


# ----------------------------------------------------------------------
# 启动恢复：repair_orphan_sessions
# ----------------------------------------------------------------------


async def test_repair_orphan_sessions_repairs_and_resets(db_factory):
    session_id = await create_session(db_factory)
    async with db_factory() as db:
        repo = Repository(db)
        await repo.update_session_status(session_id, "running")  # 崩溃残留
        await repo.append_message(session_id, "user", [{"type": "text", "text": "q"}])
        await repo.append_message(
            session_id,
            "assistant",
            [{"type": "tool_use", "id": "tu_x", "name": "echo", "input": {}}],
            stop_reason="tool_use",
        )
        # 未落 tool_result → 孤儿

    fixed = await repair_orphan_sessions(db_factory)
    assert session_id in fixed

    async with db_factory() as db:
        repo = Repository(db)
        messages = await repo.get_messages(session_id)
        row = await repo.get_session(session_id)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    synthetic = messages[2].content[0]
    assert synthetic["type"] == "tool_result"
    assert synthetic["tool_use_id"] == "tu_x"
    assert synthetic["is_error"] is True
    assert row.status == "idle"  # 非 idle 残留归位


# ----------------------------------------------------------------------
# 审批条目清理（回归 minor-3）：同 run 内复用 tool_use_id 可再次审批
# ----------------------------------------------------------------------


async def test_approval_same_tool_use_id_reusable_within_run(db_factory, loader, registry):
    """同一 run 内模型复用 tool_use_id（30 轮循环的典型形态）：
    已决议条目必须清除并可重注册新 Future，第二次审批不得静默 deny。"""
    llm = ScriptedLLM(
        [
            tool_use_turn("tu_same", "risky", {"text": "a"}),
            tool_use_turn("tu_same", "risky", {"text": "b"}),
            end_turn("finally"),
        ]
    )
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "go")

    # 第一次审批
    await wait_for_approval(manager, session_id, "tu_same")
    assert await manager.resolve_approval(session_id, "tu_same", True) is True
    # 第二次审批（同 id 重注册新 Future）
    await wait_for_approval(manager, session_id, "tu_same")
    assert await manager.resolve_approval(session_id, "tu_same", True) is True

    await wait_until(lambda: len(llm.calls) == 3)
    await wait_run_done(manager, session_id)
    await manager.stop_all()

    async with db_factory() as db:
        calls = await Repository(db).list_tool_calls(session_id)
    assert len(calls) == 2
    assert all(c.approval == "approved" for c in calls)  # 两次都是 approved，无静默 denied


# ----------------------------------------------------------------------
# 事件编号统一权威（真实模式缺陷 #2：seq 重复 / 重连雪球 / 幻影收尾）
# ----------------------------------------------------------------------


class GatedLLM:
    """挂在 user 消息落库之后、产回之前——制造「run 进行中、DB 只有
    user 消息」的真实缺陷现场。"""

    def __init__(self, payload: str) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._payload = payload

    async def run_turn(self, *, on_delta: Any = None, **kwargs: Any) -> LLMTurnResult:
        self.started.set()
        await self.release.wait()
        if on_delta is not None:
            await on_delta(StreamDelta(kind="text", text=self._payload))
        return LLMTurnResult(
            content=[{"type": "text", "text": self._payload}],
            stop_reason="end_turn",
            usage=UsageData(150, 10, 140),
        )


async def test_midrun_hydration_no_duplicate_seq(db_factory, loader, registry):
    """run 进行中（DB 仅 user 消息）首次订阅 → 推导回填 seq 从 last_seq 续编，
    后续实时事件严格续接：全流单调、无重复 seq、无幻影 0 用量收尾。"""
    gated = GatedLLM("real answer")
    manager = make_manager(db_factory, loader, registry, gated)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "hello")
    await wait_until(lambda: gated.started.is_set())  # user 已落库，run 挂在 LLM 上

    agen = manager.bus.subscribe(session_id, after_seq=0)
    # 推导回填：DB 只有 user 消息 → 恰 1 条 text_delta(user)，无合成收尾
    first = await asyncio.wait_for(anext(agen), 2.0)
    assert first.seq == 1
    assert first.type is SSEEventType.TEXT_DELTA
    assert first.data.get("role") == "user"
    assert first.data["text"] == "hello"

    # 放行 LLM → 实时事件经 sink 续编（不再从 1 重编号）
    gated.release.set()
    second = await asyncio.wait_for(anext(agen), 3.0)
    third = await asyncio.wait_for(anext(agen), 3.0)
    await agen.aclose()
    await manager.stop_all()

    seqs = [first.seq, second.seq, third.seq]
    assert seqs == [1, 2, 3]  # 严格单调、零重复
    assert second.type is SSEEventType.TEXT_DELTA and second.data["text"] == "real answer"
    assert third.type is SSEEventType.TURN_COMPLETE
    assert third.data["usage"]["input_tokens"] == 150  # 真实用量（非 0 用量幻影）


async def test_reconnect_storm_bounded_and_monotonic(db_factory, loader, registry):
    """反复 subscribe（模拟前端重连风暴）：每次重放完全一致、seq 单调不回退、
    缓冲总量有界（推导密封后不随重连次数膨胀）。"""
    llm = ScriptedLLM(
        [end_turn("done")],
        deltas=[[StreamDelta(kind="text", text="done")]],
    )
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "q")
    await wait_until(lambda: len(llm.calls) == 1)
    await wait_run_done(manager, session_id)

    baseline: list[int] = []
    baseline_len = -1
    for _ in range(5):
        agen = manager.bus.subscribe(session_id, after_seq=0)
        seqs: list[int] = []
        while True:
            try:
                ev = await asyncio.wait_for(anext(agen), 0.3)
                seqs.append(ev.seq)
            except asyncio.TimeoutError:
                break  # 重放耗尽，进入实时等待
        await agen.aclose()
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # 单调无重复
        if not baseline:
            baseline = seqs
            baseline_len = len(manager.bus._states[session_id].buffer)
        else:
            assert seqs == baseline  # 每次重放完全一致
        assert len(manager.bus._states[session_id].buffer) == baseline_len  # 总量有界
    await manager.stop_all()


async def test_derivation_sealed_once_per_process(db_factory, loader, registry):
    """推导每进程至多一次：DB 后续新增消息不再触发二次推导回填，
    重订阅的事件流与首订阅完全一致。"""
    llm = ScriptedLLM([end_turn("only run")])
    manager = make_manager(db_factory, loader, registry, llm)
    session_id = await create_session(db_factory)
    manager.enqueue_message(session_id, "q")
    await wait_until(lambda: len(llm.calls) == 1)
    await wait_run_done(manager, session_id)

    agen = manager.bus.subscribe(session_id, after_seq=0)  # 首次订阅 → 推导回填+密封
    first_seqs = []
    while True:
        try:
            ev = await asyncio.wait_for(anext(agen), 0.3)
            first_seqs.append(ev.seq)
        except asyncio.TimeoutError:
            break
    await agen.aclose()

    # DB 直接新增一条消息（绕过 run）；密封后不得二次推导
    async with db_factory() as db:
        await Repository(db).append_message(
            session_id, "assistant", [{"type": "text", "text": "injected later"}],
            stop_reason="end_turn",
        )

    agen = manager.bus.subscribe(session_id, after_seq=0)
    second_seqs = []
    while True:
        try:
            ev = await asyncio.wait_for(anext(agen), 0.3)
            second_seqs.append(ev.seq)
        except asyncio.TimeoutError:
            break
    await agen.aclose()
    await manager.stop_all()

    assert second_seqs == first_seqs  # 无二次回填
    assert len(manager.bus._states[session_id].buffer) == len(first_seqs)


async def test_repair_orphan_sessions_skips_idle_sessions(db_factory):
    session_id = await create_session(db_factory)
    async with db_factory() as db:
        repo = Repository(db)
        await repo.append_message(session_id, "user", [{"type": "text", "text": "q"}])
        # idle 状态，即使有历史也不动
    fixed = await repair_orphan_sessions(db_factory)
    assert fixed == []
