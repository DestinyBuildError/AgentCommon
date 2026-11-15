
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.agent_loop import AgentLoop, RunOutcome, SessionNotFoundError
from app.core.dispatcher import (
    DEFAULT_PERMISSION_MODE,
    ApproveCallback,
    PERMISSION_MODES,
    ToolDispatcher,
)
from app.core.event_bus import SessionEventBus
from app.core.repair import needs_repair, repair_orphans
from app.db.repository import Repository
from app.schemas.events import (
    ErrorData,
    SSEEvent,
    SSEEventType,
    TurnCompleteData,
)
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DispatcherFactory = Callable[[ApproveCallback], ToolDispatcher]


@dataclass
class PendingApproval:

    future: asyncio.Future
    tool_name: str
    claimed: bool = False


@dataclass
class PendingChoice:

    future: asyncio.Future
    claimed: bool = False


@dataclass
class RunningSession:

    session_id: str
    task: asyncio.Task
    queue: asyncio.Queue
    bus: SessionEventBus
    pending_approvals: dict[str, PendingApproval] = field(default_factory=dict)
    pending_choices: dict[str, PendingChoice] = field(default_factory=dict)
    current_run: asyncio.Task | None = None
    agent_loop: AgentLoop | None = None
    closing: bool = False


async def repair_orphan_sessions(session_factory: Any) -> list[str]:
    repaired: list[str] = []
    async with session_factory() as db:
        repo = Repository(db)
        for session in await repo.list_sessions(limit=1000):
            if session.status == "idle":
                continue
            messages = await repo.get_messages(session.id)
            history = [{"role": m.role, "content": m.content} for m in messages]
            if needs_repair(history):
                fixed = repair_orphans(history)
                inserted = fixed[len(history):]
                for entry in inserted:
                    await repo.append_message(session.id, entry["role"], entry["content"])
                repaired.append(session.id)
                logger.warning(
                    "repair_orphans: session %s got %d synthetic tool_result(s)",
                    session.id,
                    len(inserted),
                )
            await repo.update_session_status(session.id, "idle")
    return repaired


class SessionManager:

    def __init__(
        self,
        *,
        session_factory: Any,
        bus: SessionEventBus,
        llm: Any,
        registry: ToolRegistry,
        dispatcher_factory: DispatcherFactory,
        system_prompt: str = "",
        model: str | None = None,
        max_iterations: int = 30,
        max_tokens: int = 16000,
        approval_timeout_s: float = 120.0,
        idle_timeout_s: float = 600.0,
        compactor: Any | None = None,
        planner: Any | None = None,
        choice_timeout_s: float = 120.0,
        progress_interval_s: float = 60.0,
        long_running_after_s: float = 90.0,
    ) -> None:
        self.session_factory = session_factory
        self.bus = bus
        self.llm = llm
        self.registry = registry
        self.dispatcher_factory = dispatcher_factory
        self.system_prompt = system_prompt
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.approval_timeout_s = approval_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.compactor = compactor
        self.planner = planner
        self.choice_timeout_s = choice_timeout_s
        self.progress_interval_s = progress_interval_s
        self.long_running_after_s = long_running_after_s
        self.permission_mode: str = DEFAULT_PERMISSION_MODE
        self.sessions: dict[str, RunningSession] = {}


    def start_session(self, session_id: str) -> RunningSession:
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        rs = RunningSession(
            session_id=session_id,
            task=None,
            queue=asyncio.Queue(),
            bus=self.bus,
        )
        rs.task = asyncio.create_task(
            self._consume(rs), name=f"session-consumer:{session_id}"
        )
        self.sessions[session_id] = rs
        return rs

    def get_running(self, session_id: str) -> RunningSession | None:
        return self.sessions.get(session_id)

    async def stop_session(self, session_id: str) -> bool:
        rs = self.sessions.get(session_id)
        if rs is None:
            return False
        rs.closing = True
        run = rs.current_run
        if run is not None and not run.done():
            run.cancel()
        if rs.task is not None and not rs.task.done():
            rs.task.cancel()
        tasks = [t for t in (run, rs.task) if t is not None and not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.pop(session_id, None)
        return True

    async def stop_all(self) -> None:
        for session_id in list(self.sessions):
            await self.stop_session(session_id)


    def enqueue_message(self, session_id: str, text: str) -> RunningSession:
        rs = self.get_running(session_id) or self.start_session(session_id)
        rs.queue.put_nowait(text)
        return rs

    async def resolve_approval(self, session_id: str, tool_use_id: str, allow: bool) -> bool:
        rs = self.get_running(session_id)
        if rs is None:
            return False
        entry = rs.pending_approvals.get(tool_use_id)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(bool(allow))
        return True

    async def resolve_choice(self, session_id: str, choice_id: str, selected: int) -> bool:
        rs = self.get_running(session_id)
        if rs is None:
            return False
        entry = rs.pending_choices.get(choice_id)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(int(selected))
        return True


    async def set_permission_mode(self, mode: str) -> None:
        if mode not in PERMISSION_MODES:
            raise ValueError(f"invalid permission mode: {mode!r}")
        self.permission_mode = mode
        for rs in list(self.sessions.values()):
            loop = rs.agent_loop
            if loop is not None:
                loop.set_permission_mode(mode)
        try:
            async with self.session_factory() as db:
                await Repository(db).set_setting("permission_mode", mode)
        except Exception:
            logger.exception("failed to persist permission mode %r", mode)

    async def interrupt(self, session_id: str) -> bool:
        rs = self.get_running(session_id)
        if rs is None or rs.current_run is None or rs.current_run.done():
            return False
        rs.current_run.cancel()
        await asyncio.gather(rs.current_run, return_exceptions=True)
        return True


    async def _consume(self, rs: RunningSession) -> None:
        try:
            while True:
                try:
                    async with asyncio.timeout(self.idle_timeout_s):
                        text = await rs.queue.get()
                except TimeoutError:
                    if not rs.queue.empty():
                        continue
                    logger.info("session %s idle timeout; consumer exiting", rs.session_id)
                    break
                await self._run_one(rs, text)
        except asyncio.CancelledError:
            raise
        finally:
            self.sessions.pop(rs.session_id, None)

    async def _run_one(self, rs: RunningSession, text: str) -> None:
        run_task = asyncio.create_task(self._run_agent(rs, text))
        rs.current_run = run_task
        try:
            await run_task
        except asyncio.CancelledError:
            if rs.closing or asyncio.current_task().cancelling() > 0:
                raise
            self.bus.emit(
                rs.session_id,
                SSEEventType.TURN_COMPLETE,
                TurnCompleteData(stop_reason="interrupted"),
            )
        except SessionNotFoundError as exc:
            logger.error("session %s not found: %s", rs.session_id, exc)
            self.bus.emit(
                rs.session_id,
                SSEEventType.ERROR,
                ErrorData(code="session_not_found", message=str(exc)),
            )
        except Exception as exc:
            logger.exception("agent run failed for session %s", rs.session_id)
            self.bus.emit(
                rs.session_id,
                SSEEventType.ERROR,
                ErrorData(code="run_failed", message=str(exc)),
            )
        finally:
            rs.current_run = None
            self._cancel_pending_approvals(rs)

    async def _run_agent(self, rs: RunningSession, text: str) -> RunOutcome:
        async with self.session_factory() as db:
            repo = Repository(db)
            dispatcher = self.dispatcher_factory(self._make_approve_callback(rs))
            dispatcher.set_permission_mode(self.permission_mode)
            loop = AgentLoop(
                llm=self.llm,
                repo=repo,
                registry=self.registry,
                dispatcher=dispatcher,
                system_prompt=self.system_prompt,
                event_sink=self._make_event_sink(rs),
                max_iterations=self.max_iterations,
                max_tokens=self.max_tokens,
                model=self.model,
                compactor=self.compactor,
                planner=self.planner,
                choice_waiter=self._make_choice_waiter(rs),
                choice_timeout_s=int(self.choice_timeout_s),
                progress_interval_s=self.progress_interval_s,
                long_running_after_s=self.long_running_after_s,
            )
            rs.agent_loop = loop
            return await loop.run(rs.session_id, text)


    def _make_event_sink(self, rs: RunningSession) -> Callable[[SSEEvent], Any]:

        async def sink(event: SSEEvent) -> None:
            if event.type == SSEEventType.APPROVAL_REQUEST:
                tool_use_id = str(event.data.get("tool_use_id") or "")
                if tool_use_id:
                    entry = rs.pending_approvals.get(tool_use_id)
                    if entry is None or entry.future.done():
                        rs.pending_approvals[tool_use_id] = PendingApproval(
                            future=asyncio.get_running_loop().create_future(),
                            tool_name=str(event.data.get("name") or ""),
                        )
            elif event.type == SSEEventType.TASK_OPTIONS:
                choice_id = str(event.data.get("choice_id") or "")
                if choice_id:
                    rs.pending_choices[choice_id] = PendingChoice(
                        future=asyncio.get_running_loop().create_future(),
                    )
            self.bus.emit(rs.session_id, event.type, event.data)

        return sink

    def _make_approve_callback(self, rs: RunningSession) -> ApproveCallback:

        async def approve(tool_name: str, tool_input: Any) -> bool:
            entry = self._claim_approval(rs, tool_name)
            if entry is None:
                logger.warning(
                    "approval for tool %r has no pending future; denying", tool_name
                )
                return False
            try:
                return await asyncio.wait_for(entry.future, timeout=self.approval_timeout_s)
            except asyncio.TimeoutError:
                for tid, pending in list(rs.pending_approvals.items()):
                    if pending is entry:
                        rs.pending_approvals.pop(tid, None)
                logger.info("approval for tool %r timed out; auto deny", tool_name)
                return False

        return approve

    @staticmethod
    def _claim_approval(rs: RunningSession, tool_name: str) -> PendingApproval | None:
        for tid, entry in rs.pending_approvals.items():
            if not entry.claimed and (entry.tool_name == tool_name or not tool_name):
                entry.claimed = True
                if entry.future.done():
                    rs.pending_approvals.pop(tid, None)
                return entry
        return None

    @staticmethod
    def _cancel_pending_approvals(rs: RunningSession) -> None:
        for entry in rs.pending_approvals.values():
            if not entry.future.done():
                entry.future.cancel()
        rs.pending_approvals.clear()
        for entry in rs.pending_choices.values():
            if not entry.future.done():
                entry.future.cancel()
        rs.pending_choices.clear()

    def _make_choice_waiter(self, rs: RunningSession) -> Any:

        async def wait_choice(choice_id: str) -> int | None:
            entry = rs.pending_choices.get(choice_id)
            if entry is None or entry.future.done():
                rs.pending_choices.pop(choice_id, None)
                logger.warning(
                    "choice %s has no pending future; auto adopt recommended", choice_id
                )
                return None
            entry.claimed = True
            try:
                return int(
                    await asyncio.wait_for(entry.future, timeout=self.choice_timeout_s)
                )
            except asyncio.TimeoutError:
                logger.info("choice %s timed out; auto adopt recommended", choice_id)
                return None
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("choice waiter failed for %s", choice_id)
                return None
            finally:
                rs.pending_choices.pop(choice_id, None)

        return wait_choice
