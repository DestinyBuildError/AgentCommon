"""会话事件总线（/）：SessionManager ↔ SSE 桥，含缓冲重放与 DB 推导。

职责：
- 每会话一个有界缓冲区（deque，容量默认 1000，满了丢最老并记录日志）
- publish：喂给所有活跃订阅者（同步、不阻塞，绝不影响引擎循环）
- subscribe：异步生成器——先重放缓冲区中 seq > after_seq 的事件（按 seq 去重），
  再接实时流；缓冲区覆盖不到（服务重启 / 缓冲溢出 / 冷会话）时从 DB messages
  推导历史事件补齐（把 content blocks 转回对应事件类型），保证重连不丢

seq 归一策略（设计决策，真实模式缺陷 #2 后收紧）：
- **总线是 seq 的唯一权威**：所有进入缓冲的事件（实时事件由 SessionManager
  的 event_sink 经 emit() 转发、DB 推导历史由 _hydrate() 回填）都按
  seq = last_seq + 1 续编，同步赋值无 await 点、天然原子——同一会话内
  严格单调、绝不重复。AgentLoop 实例内部计数不再对 wire 编号产生影响。
- 跨进程重启：内存 seq 归零、缓冲为空；首次 subscribe 时从 DB 推导历史
  事件续编回填，live 从推导末尾接续。重启后的流是自洽的全新 seq 空间：
  客户端检测到 seq 回退 / 收到全量历史时应整体替换本地状态（P2 前端
  联动点，前端已实现 resync 语义）。delta 级事件（text_delta 中间增量）
  不落库，推导的是消息级完整内容，语义等价且幂等。
- DB 推导每进程每会话至多一次（derived_sealed 密封）：反复重连各自推导
  回填会造成事件总量随重连次数线性膨胀（雪球）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.schemas.events import (
    SSEEvent,
    SSEEventType,
    TextDeltaData,
    ThinkingDeltaData,
    ToolCallStartData,
    ToolCallResultData,
    TurnCompleteData,
    UsageOut,
    make_event,
)

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_CAPACITY = 4000
PREVIEW_LEN = 200  # 与 agent_loop.PREVIEW_LEN 对齐
# 缓冲满告警的最小间隔（秒）：长流式响应 delta 每秒几十条，缓冲满属正常
# 运行态（旧事件由 DB 推导兜底重连），逐条 WARNING 会刷屏——每会话至多
# BUFFER_WARN_INTERVAL_S 一条，期间的丢弃静默计数
BUFFER_WARN_INTERVAL_S = 5.0

# 从 DB messages 推导历史事件的加载器：async (session_id) -> list[Message-like]
# Message-like = ORM Message 或 {"role", "content", "stop_reason", ...} Mapping
HistoryLoader = Callable[[str], Awaitable[Sequence[Any]]]

# 订阅结束哨兵（close_session 时投递）
_CLOSE = object()


def _field(message: Any, name: str, default: Any = None) -> Any:
    """兼容 ORM 对象与 dict 两种消息形态。"""
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def derive_events_from_messages(messages: Sequence[Any]) -> list[SSEEvent]:
    """把 DB messages（完整 content blocks）反向推导为 SSE 事件（seq 暂置 0）。

    顺序：按消息序逐条展开——assistant 的 text/thinking → text/thinking_delta、
    tool_use → tool_call_start；user 的 text → text_delta（**携带 role="user"**，
    P3 联调遗留①：否则重启刷新后用户消息会被前端渲染成 assistant 样式）、
    tool_result → tool_call_result。仅当**最后一条消息是 assistant** 时补一条
    turn_complete（stop_reason 取该消息的 stop_reason，usage 为全部消息 token
    列累计）——run 进行中只落了 user 消息时**不合成**收尾事件（真实模式缺陷
    #2：0 用量的幻影 turn_complete 会诱导前端误判轮次结束）。空消息列表
    返回 []。
    """
    if not messages:
        return []
    events: list[SSEEvent] = []
    stop_reason: str | None = None
    usage = UsageOut()
    for message in messages:
        role = _field(message, "role")
        content = _field(message, "content") or []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text") or "")
                if text:
                    events.append(
                        make_event(
                            0,
                            SSEEventType.TEXT_DELTA,
                            TextDeltaData(
                                text=text, role=str(role) if role == "user" else None
                            ),
                        )
                    )
            elif btype == "thinking":
                text = str(block.get("thinking") or "")
                if text:
                    events.append(
                        make_event(0, SSEEventType.THINKING_DELTA, ThinkingDeltaData(text=text))
                    )
            elif btype == "tool_use":
                if block.get("id") and block.get("name"):
                    events.append(
                        make_event(
                            0,
                            SSEEventType.TOOL_CALL_START,
                            ToolCallStartData(
                                tool_use_id=str(block["id"]),
                                name=str(block["name"]),
                                input_preview=json.dumps(
                                    block.get("input") or {}, ensure_ascii=False
                                )[:PREVIEW_LEN],
                            ),
                        )
                    )
            elif btype == "tool_result":
                events.append(
                    make_event(
                        0,
                        SSEEventType.TOOL_CALL_RESULT,
                        ToolCallResultData(
                            tool_use_id=str(block.get("tool_use_id") or ""),
                            is_error=bool(block.get("is_error", False)),
                            output_preview=str(block.get("content") or "")[:PREVIEW_LEN],
                        ),
                    )
                )
        if role == "assistant":
            sr = _field(message, "stop_reason")
            if sr:
                stop_reason = str(sr)
        usage = UsageOut(
            input_tokens=usage.input_tokens + int(_field(message, "input_tokens", 0) or 0),
            output_tokens=usage.output_tokens + int(_field(message, "output_tokens", 0) or 0),
            cache_read=usage.cache_read + int(_field(message, "cache_read", 0) or 0),
            cache_creation=usage.cache_creation + int(_field(message, "cache_creation", 0) or 0),
        )
    last_role = _field(messages[-1], "role")
    if last_role == "assistant":
        events.append(
            make_event(
                0,
                SSEEventType.TURN_COMPLETE,
                TurnCompleteData(stop_reason=stop_reason or "end_turn", usage=usage, cost_usd=0.0),
            )
        )
    return events


@dataclass
class _SessionState:
    """单个会话的总线状态：有界缓冲 + 活跃订阅者 + seq 游标。"""

    buffer: deque[SSEEvent] = field(default_factory=deque)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    last_seq: int = 0
    # DB 推导回填每进程每会话至多一次（真实模式缺陷 #2：反复重连各自推导
    # 回填是事件雪球放大的根源之一）；loader 失败不置位，允许下次重试
    derived_sealed: bool = False
    # 缓冲满告警节流（实测：长流式 delta 洪峰逐条 WARNING 刷屏）
    last_buffer_warn: float = 0.0


def _push(queue: asyncio.Queue, item: Any) -> None:
    """非阻塞投递；订阅者队列满时丢最老保最新（客户端可凭 seq 缺口请求重放）。"""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover - 理论不可达
            pass


class SessionEventBus:
    """进程内会话事件总线：发布 / 有界缓冲 / 重放 / DB 推导 / 实时订阅。"""

    def __init__(
        self,
        history_loader: HistoryLoader | None = None,
        *,
        capacity: int = DEFAULT_BUFFER_CAPACITY,
    ) -> None:
        self._loader = history_loader
        self._capacity = capacity
        self._states: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # 发布侧
    # ------------------------------------------------------------------

    def publish(self, session_id: str, event: SSEEvent) -> None:
        """发布一条事件：入会话缓冲（满了丢最老）并推给所有活跃订阅者。

        缓冲满是流式长响应的正常运行态：在订客户端早已实时收到旧事件，
        丢弃只影响"极端重连场景的精确重放"，而重放缺口由 DB 推导全量
        重同步兜底——因此告警按会话节流（默认 5s 一条），不逐条刷屏。
        """
        state = self._state(session_id)
        if state.buffer.maxlen is not None and len(state.buffer) >= state.buffer.maxlen:
            now = time.monotonic()
            if now - state.last_buffer_warn >= BUFFER_WARN_INTERVAL_S:
                logger.warning(
                    "event buffer full for session %s; dropping oldest event seq=%s "
                    "(further drops suppressed for %ds)",
                    session_id,
                    state.buffer[0].seq,
                    int(BUFFER_WARN_INTERVAL_S),
                )
                state.last_buffer_warn = now
        state.buffer.append(event)
        state.last_seq = max(state.last_seq, event.seq)
        for queue in list(state.subscribers):
            _push(queue, event)

    def emit(self, session_id: str, event_type: SSEEventType, data: Any) -> SSEEvent:
        """总线侧统一编号发布：seq = last_seq + 1，同步赋值无 await 点，
        与 _hydrate 的推导回填天然互斥（同会话内绝不重号）。

        这是**会话 seq 的唯一权威入口**：SessionManager 的 event_sink 把
        AgentLoop 发出的全部实时事件经此转发（覆盖 loop 内部计数），
        interrupted 收尾 / 兜底 error 事件也走同一条路。
        """
        event = make_event(self.last_seq(session_id) + 1, event_type, data)
        self.publish(session_id, event)
        return event

    def last_seq(self, session_id: str) -> int:
        """会话当前末尾 seq（0 = 本进程未发布过）。"""
        state = self._states.get(session_id)
        return state.last_seq if state is not None else 0

    def known_session_ids(self) -> list[str]:
        """本进程内有总线状态的会话 id（广播用）。

        包含两类：发过消息的会话（SessionManager.start_session 后经 emit
        产生状态）与**冷会话**（创建后未发消息、但已有 SSE 订阅者——
        subscribe 经 _state 产生状态）。全局事件（workspace_changed）按此
        清单广播，保证只挂了 SSE 流的冷会话也能实时收到；向无订阅者的
        会话广播只是入缓冲（有界 deque），供其重连时 after_seq 重放补齐。
        """
        return list(self._states)

    # ------------------------------------------------------------------
    # 订阅侧
    # ------------------------------------------------------------------

    async def subscribe(
        self, session_id: str, after_seq: int = 0
    ) -> AsyncIterator[SSEEvent]:
        """订阅会话事件流：先重放 after_seq 之后的事件（去重），再接实时流。

        - 缓冲区能覆盖游标缺口 → 直接重放（精确、同 seq 空间、零重复，
          严格只发 seq > after_seq）
        - 游标超出本进程 seq 空间（跨重启旧游标）或与缓冲起点间存在缺口
          （缓冲溢出丢最老）→ 全量重同步：从 DB 推导历史补齐（每进程至多
          一次，seq 续编不与实时段重叠）后全量重放（客户端按 seq 幂等去重；
          详见模块 docstring 的 seq 归一策略）
        - close_session 会让所有订阅者以 StopAsyncIteration 结束
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._capacity)
        async with self._lock(session_id):
            state = self._state(session_id)
            # 必须在 hydrate 之前判定：缺口判定依赖 pre-hydrate 的缓冲起点
            resync = self._needs_resync(state, after_seq)
            if not self._covers(state, after_seq):
                await self._hydrate(session_id, state)
            for event in self._replay(state, after_seq, resync):
                queue.put_nowait(event)
            state.subscribers.add(queue)
        try:
            while True:
                item = await queue.get()
                if item is _CLOSE:
                    break
                assert isinstance(item, SSEEvent)
                yield item
        finally:
            state = self._states.get(session_id)
            if state is not None:
                state.subscribers.discard(queue)

    def close_session(self, session_id: str) -> None:
        """丢弃会话总线状态并结束全部订阅者（DELETE 会话 / 进程收尾调用）。"""
        self._locks.pop(session_id, None)
        state = self._states.pop(session_id, None)
        if state is None:
            return
        for queue in list(state.subscribers):
            _push(queue, _CLOSE)
        state.subscribers.clear()

    # ------------------------------------------------------------------

    def _state(self, session_id: str) -> _SessionState:
        state = self._states.get(session_id)
        if state is None:
            state = _SessionState()
            state.buffer = deque(maxlen=self._capacity)
            self._states[session_id] = state
        return state

    def _lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def _covers(self, state: _SessionState, after_seq: int) -> bool:
        """缓冲区能否精确覆盖 [after_seq, last_seq] 的缺口。"""
        if state.last_seq == 0:
            return False  # 本进程从未发布 → 尝试 DB 推导（重启 / 冷会话）
        if after_seq >= state.last_seq:
            return True  # 已追平：无需重放
        return bool(state.buffer) and state.buffer[0].seq <= after_seq + 1

    def _needs_resync(self, state: _SessionState, after_seq: int) -> bool:
        """是否需要全量重同步（在 hydrate 之前、基于 pre-hydrate 缓冲判定）。

        - 游标超出本进程 seq 空间（跨重启的旧游标，含冷会话上的旧游标）
        - 游标与缓冲起点之间存在缺口（缓冲溢出丢最老 / 缓冲为空但本进程
          已发布到更新游标之外）→ 历史不可由缓冲精确补齐，改走 DB 全量重同步
        """
        if after_seq > state.last_seq:
            return True
        if state.buffer:
            return state.buffer[0].seq > after_seq + 1
        return state.last_seq > after_seq

    async def _hydrate(self, session_id: str, state: _SessionState) -> None:
        """缓冲覆盖不到时：从 DB messages 推导历史事件，seq 从 last_seq 续编
        （不与已有实时段重叠），**追加**到缓冲尾部（保持 seq 升序）。

        编号规则是本函数的正确性关键（真实模式缺陷 #2）：推导事件绝不能
        重新从 1 编号——run 进行中订阅时，AgentLoop 经 event_sink 由总线
        统一编号的实时事件可能已占用 1..M；再从 1 编号必然产生重复 seq。
        续编后与 `emit()` 的统一编号天然衔接。

        每进程每会话只推导一次（成功或空结果都置 derived_sealed）：
        反复重连各自推导回填是事件雪球的另一根源。loader 异常不置位，
        允许下次订阅重试。
        """
        if self._loader is None or state.derived_sealed:
            return
        try:
            messages = await self._loader(session_id)
        except Exception:
            logger.exception("history loader failed for session %s", session_id)
            return
        state.derived_sealed = True  # 推导至多一次（含空结果）
        derived = derive_events_from_messages(messages)
        if not derived:
            return
        seq = state.last_seq
        fresh: list[SSEEvent] = []
        for base in derived:
            seq += 1
            fresh.append(make_event(seq, base.type, base.data))
        # 追加到尾部：fresh.seq 全部 > 缓冲中已有事件 seq，保持升序
        state.buffer.extend(fresh)
        state.last_seq = seq
        logger.info(
            "hydrated %d history events for session %s (through seq=%d)",
            len(fresh),
            session_id,
            seq,
        )

    def _replay(
        self, state: _SessionState, after_seq: int, resync: bool
    ) -> list[SSEEvent]:
        """按订阅游标计算重放列表（保持 seq 升序、去重）。

        - resync=False（游标在本进程 seq 空间内且缓冲无缺口）：严格只重放
          seq > after_seq——"after_seq=N 只回 N 之后"的精确语义
        - resync=True（跨重启旧游标 / 缓冲缺口）：全量重放缓冲（推导段 +
          实时段），客户端按 seq 幂等去重
        - 已追平（after_seq >= last_seq 且无缺口）→ 不重放，只等实时
        """
        seen: set[int] = set()
        out: list[SSEEvent] = []
        for event in state.buffer:
            if event.seq in seen:
                continue
            seen.add(event.seq)
            if resync or event.seq > after_seq:
                out.append(event)
        return out
