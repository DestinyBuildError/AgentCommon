"""并发会话管理：会话任务生命周期、消息队列、审批 Future、启动恢复。

设计要点：
- 一个会话同时只有一个 loop 消费者——用户连发多条消息进 queue，按序消费，
  天然避免历史并发写坏
- 每个 run 构造一个 AgentLoop（llm / registry / 会话级 seq 复用），事件编号
  从 event_bus 记录的会话末尾 seq 接续——会话级 seq 单调连贯（跨 run / 跨
  loop 实例；跨进程重启由 event_bus 的 DB 推导负责）
- 空闲超时（config.session_idle_timeout_s，默认 600s）自动停止消费者任务；
  历史在 DB，随时重新唤起
- 审批门：approval_request 事件到达时注册 Future（引擎经 event_sink 通知，
  事件携带 tool_use_id）；dispatcher 的 approve_callback 挂起等待；
  resolve_approval 设置 Future 结果；超时 config.approval_timeout_s
  （默认 120s）自动 deny
- interrupt：取消当前 run 任务；引擎已在 finally 合成 interrupted
  tool_result 并落库（本层不再重复），本层补发
  turn_complete(stop_reason="interrupted") 供前端收尾
"""

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

# app 层注入：approve_callback → 每会话独立的 ToolDispatcher
# （AgentLoop.__init__ 会 set_approval_notifier，dispatcher 不可跨会话共享）
DispatcherFactory = Callable[[ApproveCallback], ToolDispatcher]


@dataclass
class PendingApproval:
    """一个待审批的危险工具调用（键：tool_use_id）。"""

    future: asyncio.Future
    tool_name: str
    claimed: bool = False  # 已被 approve_callback 认领（防止并发重复认领）


@dataclass
class PendingChoice:
    """一个待用户选择的任务拆解方案（键：choice_id）。

    与审批门同构：event_sink 看到 task_options 事件即注册 Future；
    resolve_choice（POST /choices）设置结果；choice_waiter 认领并
    wait_for，超时返回 None → AgentLoop 自动采用推荐方案。
    """

    future: asyncio.Future
    claimed: bool = False  # 已被 choice_waiter 认领（防重复认领）


@dataclass
class RunningSession:
    """会话的运行时句柄（RunningSession）。"""

    session_id: str
    task: asyncio.Task                       # 消费循环任务（一个会话一个）
    queue: asyncio.Queue                     # 待处理用户消息
    bus: SessionEventBus                     # 事件总线（进程级共享实例）
    pending_approvals: dict[str, PendingApproval] = field(default_factory=dict)
    pending_choices: dict[str, PendingChoice] = field(default_factory=dict)
    current_run: asyncio.Task | None = None  # 正在执行的 AgentLoop.run 任务
    agent_loop: AgentLoop | None = None      # 当前 run 的 loop（观测用）
    closing: bool = False                    # 消费者被 stop/shutdown 取消中


# ----------------------------------------------------------------------
# 启动恢复（：服务启动时对所有未完成会话执行 repair_orphans）
# ----------------------------------------------------------------------


async def repair_orphan_sessions(session_factory: Any) -> list[str]:
    """对所有非 idle 状态会话做孤儿 tool_use 修复并回写 DB，状态归位 idle。

    崩溃残留的 running 状态会话即使无孤儿也要归位（进程重启后不再有人跑它）。
    返回实际插入过合成 tool_result 的会话 id 列表。
    """
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
    """并发会话生命周期管理（llm / registry / dispatcher 由 app 层注入）。"""

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
        self.compactor = compactor  # Compactor；None → AgentLoop 不做 compaction
        # 任务拆解规划器；None → AgentLoop 跳过规划直接进循环
        self.planner = planner
        self.choice_timeout_s = choice_timeout_s
        # 长时任务进度推送参数，逐 run 透传给 AgentLoop
        self.progress_interval_s = progress_interval_s
        self.long_running_after_s = long_running_after_s
        # 全局权限模式：运行时权威；持久化在 app_settings 表，
        # lifespan 启动时装载。dispatcher 每 run 创建时取此值，切换经
        # set_permission_mode 传播到运行中的 AgentLoop
        self.permission_mode: str = DEFAULT_PERMISSION_MODE
        self.sessions: dict[str, RunningSession] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start_session(self, session_id: str) -> RunningSession:
        """启动会话消费者任务（幂等：已运行则原样返回）。"""
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        rs = RunningSession(
            session_id=session_id,
            task=None,  # type: ignore[arg-type]
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
        """停止会话：取消当前 run（引擎合成 interrupted 结果落库）与消费者任务。
        不存在返回 False。总线缓冲保留——重连客户端仍可按 seq 重放。"""
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
        """进程收尾：停掉全部会话任务（main.lifespan shutdown 调用）。"""
        for session_id in list(self.sessions):
            await self.stop_session(session_id)

    # ------------------------------------------------------------------
    # 入口：发消息 / 审批 / 打断
    # ------------------------------------------------------------------

    def enqueue_message(self, session_id: str, text: str) -> RunningSession:
        """入队用户消息并确保消费者任务在跑（不在则唤起）。"""
        rs = self.get_running(session_id) or self.start_session(session_id)
        rs.queue.put_nowait(text)
        return rs

    async def resolve_approval(self, session_id: str, tool_use_id: str, allow: bool) -> bool:
        """审批回调：设置对应 Future 结果。无此 pending（或已超时/已结束）
        返回 False（API 层据此 404）。"""
        rs = self.get_running(session_id)
        if rs is None:
            return False
        entry = rs.pending_approvals.get(tool_use_id)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(bool(allow))
        return True

    async def resolve_choice(self, session_id: str, choice_id: str, selected: int) -> bool:
        """任务拆解选择回调：设置对应 Future 结果。

        无此 pending（已超时自动采用/会话不在跑）返回 False（API 层据此 404）；
        迟到决议按 approval 同语义处理——wait_for 已取消 Future 时这里 404。
        """
        rs = self.get_running(session_id)
        if rs is None:
            return False
        entry = rs.pending_choices.get(choice_id)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(int(selected))
        return True

    # ------------------------------------------------------------------
    # 全局权限模式
    # ------------------------------------------------------------------

    async def set_permission_mode(self, mode: str) -> None:
        """切换全局权限模式：内存生效 → 持久化 → 传播到运行中的 dispatcher。

        传播顺序故意"先内存后落库"：权限收紧要求即时生效（落库失败只影响
        重启恢复，不影响本次运行的判定）；传播到运行中 AgentLoop 是尽力而
        为（无运行会话时静默跳过）。非法模式在 dispatcher 侧抛 ValueError。
        """
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
        """打断当前 run：取消 run 任务并等待引擎完成 interrupted 落库收尾。
        会话不在跑或当前无 run 返回 False。"""
        rs = self.get_running(session_id)
        if rs is None or rs.current_run is None or rs.current_run.done():
            return False
        rs.current_run.cancel()
        await asyncio.gather(rs.current_run, return_exceptions=True)
        return True

    # ------------------------------------------------------------------
    # 消费循环（一个会话同时只有一个）
    # ------------------------------------------------------------------

    async def _consume(self, rs: RunningSession) -> None:
        """按序消费队列中的用户消息；空闲超时自动退出。"""
        try:
            while True:
                try:
                    async with asyncio.timeout(self.idle_timeout_s):
                        text = await rs.queue.get()
                except TimeoutError:
                    if not rs.queue.empty():
                        continue  # 超时瞬间恰好入队的竞态兜底
                    logger.info("session %s idle timeout; consumer exiting", rs.session_id)
                    break
                await self._run_one(rs, text)
        except asyncio.CancelledError:
            raise  # stop_session / shutdown：由调用方 gather 收尾
        finally:
            self.sessions.pop(rs.session_id, None)

    async def _run_one(self, rs: RunningSession, text: str) -> None:
        run_task = asyncio.create_task(self._run_agent(rs, text))
        rs.current_run = run_task
        try:
            await run_task
        except asyncio.CancelledError:
            # 区分两类取消：
            # - 消费者任务自身被取消（stop_session / 事件循环关停 _cancel_all_tasks）：
            # current_task().cancelling() > 0 → 必须原样上抛让任务消亡；
            # 若吞掉（误判为 interrupt），关停时的 gather 会等它直到空闲超时
            # （默认 600s）——表现为"进程收尾/测试收尾挂死"。
            # - 仅子 run 任务被取消（interrupt()）：消费者自身无未决取消请求，
            # 按 中断处理——引擎已在 finally 合成 interrupted 结果并落库，
            # 这里补发收尾事件供前端停止"正在输入"态。
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
        except Exception as exc:  # 引擎已归一化大部分失败，这里兜底防消费者死亡
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
        """构造（每 run 一个）AgentLoop 并执行；llm/registry/seq 复用会话级状态。"""
        async with self.session_factory() as db:
            repo = Repository(db)
            dispatcher = self.dispatcher_factory(self._make_approve_callback(rs))
            # 每 run 的 dispatcher 按当前全局权限模式判定
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
            # wire seq 由 event_sink 经 bus.emit 统一编号（见 _make_event_sink），
            # loop 实例内部计数不再参与 wire 编号
            rs.agent_loop = loop
            return await loop.run(rs.session_id, text)

    # ------------------------------------------------------------------
    # 审批门
    # ------------------------------------------------------------------

    def _make_event_sink(self, rs: RunningSession) -> Callable[[SSEEvent], Any]:
        """引擎事件 → 审批/选择 Future 注册 → 总线统一编号发布。

        编号以总线为唯一权威（真实模式缺陷 #2）：loop 内部计数被覆盖，
        wire seq 一律取 bus.last_seq + 1——与 _hydrate 的推导回填共用同一
        游标，任何"订阅先于首事件 / 推导回填先于 run 完成"的交错下 seq
        都严格单调不重复。
        """

        async def sink(event: SSEEvent) -> None:
            if event.type == SSEEventType.APPROVAL_REQUEST:
                tool_use_id = str(event.data.get("tool_use_id") or "")
                if tool_use_id:
                    entry = rs.pending_approvals.get(tool_use_id)
                    if entry is None or entry.future.done():
                        # 无条目或已决议 → 重注册（模型复用 tool_use_id 时审批仍生效）
                        rs.pending_approvals[tool_use_id] = PendingApproval(
                            future=asyncio.get_running_loop().create_future(),
                            tool_name=str(event.data.get("name") or ""),
                        )
            elif event.type == SSEEventType.TASK_OPTIONS:
                # 任务拆解选择门：task_options 事件到达即注册
                # PendingChoice，resolve_choice（POST /choices）据此 resolve
                choice_id = str(event.data.get("choice_id") or "")
                if choice_id:
                    rs.pending_choices[choice_id] = PendingChoice(
                        future=asyncio.get_running_loop().create_future(),
                    )
            self.bus.emit(rs.session_id, event.type, event.data)

        return sink

    def _make_approve_callback(self, rs: RunningSession) -> ApproveCallback:
        """dispatcher 审批回调：认领 pending Future 并挂起等待用户决议/超时。"""

        async def approve(tool_name: str, tool_input: Any) -> bool:
            entry = self._claim_approval(rs, tool_name)
            if entry is None:
                # 无注册中的 Future（事件管道异常等）→ 保守拒绝，危险工具不放行
                logger.warning(
                    "approval for tool %r has no pending future; denying", tool_name
                )
                return False
            try:
                return await asyncio.wait_for(entry.future, timeout=self.approval_timeout_s)
            except asyncio.TimeoutError:
                # ：审批超时自动 deny（wait_for 已取消 Future → 迟到决议 404）
                for tid, pending in list(rs.pending_approvals.items()):
                    if pending is entry:
                        rs.pending_approvals.pop(tid, None)
                logger.info("approval for tool %r timed out; auto deny", tool_name)
                return False

        return approve

    @staticmethod
    def _claim_approval(rs: RunningSession, tool_name: str) -> PendingApproval | None:
        """按名字 + FIFO 认领一个审批（并行危险工具各自认领各自的）。

        已被早到决议（future.done 且未认领）的条目同样可认领——认领后
        wait_for 立即返回其结果；同时把该条目从字典清除，保证同
        tool_use_id 后续审批请求能重新注册新 Future。
        """
        for tid, entry in rs.pending_approvals.items():
            if not entry.claimed and (entry.tool_name == tool_name or not tool_name):
                entry.claimed = True
                if entry.future.done():
                    rs.pending_approvals.pop(tid, None)
                return entry
        return None

    @staticmethod
    def _cancel_pending_approvals(rs: RunningSession) -> None:
        """run 结束后清理残留审批与未决选择（中断/异常路径的 Future）。"""
        for entry in rs.pending_approvals.values():
            if not entry.future.done():
                entry.future.cancel()
        rs.pending_approvals.clear()
        for entry in rs.pending_choices.values():
            if not entry.future.done():
                entry.future.cancel()
        rs.pending_choices.clear()

    def _make_choice_waiter(self, rs: RunningSession) -> Any:
        """任务拆解选择等待器：按 choice_id 认领 PendingChoice
        并挂起等待用户决议；超时/通道异常返回 None（AgentLoop 据此自动
        采用推荐方案，不阻塞主循环）。"""

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
                # 超时自动采用推荐（与审批超时 deny 同语义：wait_for 已取消
                # Future → 迟到的 POST /choices 拿到 404）
                logger.info("choice %s timed out; auto adopt recommended", choice_id)
                return None
            except asyncio.CancelledError:
                raise  # interrupt / stop_session：让引擎的 interrupted 收尾照常
            except Exception:
                logger.exception("choice waiter failed for %s", choice_id)
                return None
            finally:
                rs.pending_choices.pop(choice_id, None)  # 决议/超时/异常统一回收

        return wait_choice
