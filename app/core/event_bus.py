
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
PREVIEW_LEN = 200
BUFFER_WARN_INTERVAL_S = 5.0

HistoryLoader = Callable[[str], Awaitable[Sequence[Any]]]

_CLOSE = object()


def _field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def derive_events_from_messages(messages: Sequence[Any]) -> list[SSEEvent]:
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

    buffer: deque[SSEEvent] = field(default_factory=deque)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    last_seq: int = 0
    derived_sealed: bool = False
    last_buffer_warn: float = 0.0


def _push(queue: asyncio.Queue, item: Any) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass


class SessionEventBus:

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


    def publish(self, session_id: str, event: SSEEvent) -> None:
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
        event = make_event(self.last_seq(session_id) + 1, event_type, data)
        self.publish(session_id, event)
        return event

    def last_seq(self, session_id: str) -> int:
        state = self._states.get(session_id)
        return state.last_seq if state is not None else 0

    def known_session_ids(self) -> list[str]:
        return list(self._states)


    async def subscribe(
        self, session_id: str, after_seq: int = 0
    ) -> AsyncIterator[SSEEvent]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._capacity)
        async with self._lock(session_id):
            state = self._state(session_id)
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
        self._locks.pop(session_id, None)
        state = self._states.pop(session_id, None)
        if state is None:
            return
        for queue in list(state.subscribers):
            _push(queue, _CLOSE)
        state.subscribers.clear()


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
        if state.last_seq == 0:
            return False
        if after_seq >= state.last_seq:
            return True
        return bool(state.buffer) and state.buffer[0].seq <= after_seq + 1

    def _needs_resync(self, state: _SessionState, after_seq: int) -> bool:
        if after_seq > state.last_seq:
            return True
        if state.buffer:
            return state.buffer[0].seq > after_seq + 1
        return state.last_seq > after_seq

    async def _hydrate(self, session_id: str, state: _SessionState) -> None:
        if self._loader is None or state.derived_sealed:
            return
        try:
            messages = await self._loader(session_id)
        except Exception:
            logger.exception("history loader failed for session %s", session_id)
            return
        state.derived_sealed = True
        derived = derive_events_from_messages(messages)
        if not derived:
            return
        seq = state.last_seq
        fresh: list[SSEEvent] = []
        for base in derived:
            seq += 1
            fresh.append(make_event(seq, base.type, base.data))
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
        seen: set[int] = set()
        out: list[SSEEvent] = []
        for event in state.buffer:
            if event.seq in seen:
                continue
            seen.add(event.seq)
            if resync or event.seq > after_seq:
                out.append(event)
        return out
