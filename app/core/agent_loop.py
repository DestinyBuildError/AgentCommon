"""Agent 循环状态机。

    IDLE ──用户输入──▶ LLM_CALL ──stop_reason──┐
     ▲ │ │
     │ ┌──────┴─────────┐ │
     │ │ tool_use │ end_turn/refusal → IDLE
     │ ▼ │ max_tokens → 回滚快照重试（≤2 次）
     │ TOOL_EXEC（含审批门） │
     └───────────────────────────────┘

两条正确性不变量：
  1 无孤儿 tool_use：每轮结束前校验，缺结果当场合成 interrupted 结果持久化
  2 消息 append-only：轮内只追加；max_tokens 重试通过内存快照回滚重来，
    不修改任何已持久化历史（半截 assistant 消息永不落库）

对外接口（API 层 / SessionManager 波次依赖）：
    loop = AgentLoop(llm=..., repo=..., dispatcher=..., registry=...,
                     system_prompt=..., event_sink=async (SSEEvent) -> None, ...)
    outcome = await loop.run(session_id, user_text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.core.context import COMPACTION_THRESHOLD, TokenLedger
from app.core.dispatcher import ToolDispatcher, ToolResult
from app.core.llm_client import LLMFatalError, UsageData, build_system_blocks
from app.core.planner import COMPLEXITY_LABELS, PlanResult
from app.core.pricing import cost_usd
from app.core.repair import needs_repair, repair_orphans, synthetic_tool_result
from app.schemas.events import (
    ApprovalRequestData,
    ErrorData,
    SSEEvent,
    SSEEventType,
    TaskOptionData,
    TaskOptionsData,
    TaskProgressData,
    TextDeltaData,
    ThinkingDeltaData,
    ToolCallResultData,
    ToolCallStartData,
    TurnCompleteData,
    UsageOut,
    make_event,
)
from app.db.repository import Repository
from app.tools.registry import ToolRegistry

MAX_TOKENS_RETRIES = 2   # ：max_tokens 截断回滚重试 ≤2 次
PREVIEW_LEN = 200        # SSE input_preview / output_preview 截断长度

logger = logging.getLogger(__name__)

EventSink = Callable[[SSEEvent], Awaitable[None]]
# 选择等待器（SessionManager 注入）：按 choice_id 认领 PendingChoice Future
# 并挂起等待；超时/通道异常返回 None（AgentLoop 自动采用推荐方案）
ChoiceWaiter = Callable[[str], Awaitable[int | None]]


class SessionNotFoundError(Exception):
    """run() 指定的会话不存在。"""


@dataclass
class RunOutcome:
    """一次 run() 的收尾摘要（SessionManager / API 层消费）。"""

    session_id: str
    ok: bool
    stop_reason: str | None = None      # end_turn / refusal / max_iterations / error / ...
    iterations: int = 0
    error_code: str | None = None       # 非 ok 时的错误码
    usage: UsageData = field(default_factory=UsageData)  # 本次 run 累计用量
    cost_usd: float = 0.0               # 本次 run 累计成本（定价表）


class AgentLoop:
    """会话级 Agent 状态机（一个会话一个实例，事件 seq 随实例单调递增）。"""

    def __init__(
        self,
        *,
        llm: Any,
        repo: Repository,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        system_prompt: str = "",
        event_sink: EventSink | None = None,
        max_iterations: int = 30,
        max_tokens: int = 16000,
        model: str | None = None,
        compaction_threshold: int = COMPACTION_THRESHOLD,
        compactor: Any | None = None,
        planner: Any | None = None,
        choice_waiter: ChoiceWaiter | None = None,
        choice_timeout_s: int = 120,
        progress_interval_s: float = 60.0,
        long_running_after_s: float = 90.0,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._registry = registry
        self._dispatcher = dispatcher
        self._system_blocks = build_system_blocks(system_prompt) if system_prompt else None
        self._event_sink = event_sink
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._model = model
        self._compaction_threshold = compaction_threshold
        self._compactor = compactor  # Compactor；None → 不做 compaction
        # 任务拆解规划器；None → 消息直接进循环（测试/降级开关）
        self._planner = planner
        self._choice_waiter = choice_waiter
        self._choice_timeout_s = choice_timeout_s
        # 长时任务进度推送：规划器标记或运行超阈值自动升级后，
        # ticker 每 progress_interval_s 发一次 task_progress（不落库）
        # 下限 0.05s 仅防误传 0/负值死循环；生产值由 config 提供（60/90）
        self._progress_interval_s = max(0.05, float(progress_interval_s))
        self._long_running_after_s = max(0.05, float(long_running_after_s))
        self._long_running = False
        self._run_started = 0.0
        self._iteration = 0
        self._last_activity = "开始处理"
        self._progress_task: asyncio.Task | None = None
        # 审批通知 → approval_request SSE（dispatcher 不懂 SSE，钩子由 loop 接线）
        dispatcher.set_approval_notifier(self._approval_notifier())
        # SSE 事件 seq：实例内单调递增，仅作 loop 内部事件序号；
        # wire 编号以 event_bus 为唯一权威（SessionManager 的 event_sink 经
        # bus.emit 统一续编，跨 run / 跨实例 / 跨推导回填都不重号）
        self._event_seq = 0

    def set_permission_mode(self, mode: str) -> None:
        """透传全局权限模式切换（SessionManager → 运行中 dispatcher，
        下一个工具调用即按新模式判定）。"""
        self._dispatcher.set_permission_mode(mode)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def run(self, session_id: str, user_text: str) -> RunOutcome:
        session = await self._repo.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: {session_id}")
        model = session.model or self._model or "claude-opus-4-8"

        await self._repo.update_session_status(session_id, "running")
        await self._repo.append_message(
            session_id, "user", [{"type": "text", "text": user_text}]
        )
        rows = await self._repo.get_messages(session_id)
        history: list[dict[str, Any]] = [
            {"role": row.role, "content": [dict(b) for b in row.content]} for row in rows
        ]

        # ---- 任务拆解：medium/complex 先让用户选方向再执行 ----
        # 规划器永不抛异常（内部降级）；用户选择 / 超时自动采用推荐的结果
        # 以 user 注记落库 + 进内存镜像，主循环与重启恢复路径看到的一致
        plan = await self._plan_and_ask(session_id, user_text, history)
        # 长时任务：规划器扫描标记 → 立即启用分钟级进度；
        # 未标记则由 ticker 在运行超 long_running_after_s 时自动升级
        self._long_running = bool(plan is not None and plan.long_running)
        self._run_started = time.monotonic()
        self._last_activity = "开始处理" if plan is None or not plan.progress_label else (
            f"开始：{plan.progress_label}"
        )
        self._progress_task = asyncio.create_task(
            self._progress_ticker(session_id), name=f"progress-ticker:{session_id}"
        )

        ledger = TokenLedger(self._compaction_threshold)
        outcome = RunOutcome(session_id=session_id, ok=False)
        max_tokens = self._max_tokens
        max_tokens_retries = 0
        run_cost = 0.0          # 本次 run 累计成本（定价表）
        self._compacted_once = False  # compaction 每 run 至多一次（防抖）
        status = "idle"
        try:
            for iteration in range(1, self._max_iterations + 1):
                outcome.iterations = iteration
                self._iteration = iteration
                self._last_activity = "模型思考中"
                snapshot = [
                    {"role": m["role"], "content": list(m["content"])} for m in history
                ]

                # ---- LLM_CALL（流式；异常分类见 llm_client）----
                try:
                    turn = await self._llm.run_turn(
                        messages=_merge_consecutive(history),
                        system=self._system_blocks,
                        tools=self._registry.api_schemas(),
                        model=model,
                        max_tokens=max_tokens,
                        on_delta=self._delta_emitter(),
                    )
                except asyncio.CancelledError:
                    raise
                except LLMFatalError as exc:  # 认证/权限/请求非法 → 终态，不重试
                    await self._emit_error(exc.code, str(exc))
                    outcome.ok, outcome.stop_reason = False, "error"
                    outcome.error_code = exc.code
                    break
                except Exception as exc:  # 429/5xx/连接错误：SDK 已重试耗尽 → 上抛归一化
                    await self._emit_error("llm_error", f"LLM call failed: {exc}")
                    outcome.ok, outcome.stop_reason = False, "error"
                    outcome.error_code = "llm_error"
                    break

                # ---- 用量记账（含被 max_tokens 丢弃的轮次——token 已真实消耗）----
                ledger.add(turn.usage)
                turn_cost = cost_usd(
                    model,
                    input_tokens=turn.usage.input_tokens,
                    output_tokens=turn.usage.output_tokens,
                    cache_read_tokens=turn.usage.cache_read,
                    cache_creation_tokens=turn.usage.cache_creation,
                )
                run_cost += turn_cost
                await self._repo.update_session_usage(
                    session_id,
                    input_tokens=turn.usage.input_tokens,
                    output_tokens=turn.usage.output_tokens,
                    cache_read=turn.usage.cache_read,
                    cost_usd=turn_cost,
                )
                # ---- compaction：接近阈值时压缩，失败不阻塞主循环 ----
                if ledger.should_compact():
                    if await self._maybe_compact(session_id, ledger, history, model):
                        # 快照刷新：compaction 已追加 assistant 消息并落库，
                        # max_tokens 回滚必须保留它（只丢弃本轮半截 assistant）
                        snapshot = [
                            {"role": m["role"], "content": list(m["content"])}
                            for m in history
                        ]

                stop_reason = turn.stop_reason

                # ---- max_tokens：回滚快照重试（不变量 2：半截消息不落库）----
                if stop_reason == "max_tokens":
                    if max_tokens_retries < MAX_TOKENS_RETRIES:
                        max_tokens_retries += 1
                        max_tokens *= 2
                        history = snapshot  # 丢弃本轮半截 assistant，从快照重来
                        continue
                    await self._emit_error(
                        "max_tokens_exhausted",
                        f"Output truncated at max_tokens; aborted after "
                        f"{MAX_TOKENS_RETRIES} retries with doubled budget",
                    )
                    outcome.ok, outcome.stop_reason = False, "max_tokens"
                    outcome.error_code = "max_tokens_exhausted"
                    break

                # ---- end_turn / refusal / 其他终态：持久化最终消息并收尾 ----
                assistant_msg = await self._repo.append_message(
                    session_id,
                    "assistant",
                    turn.content,
                    stop_reason=stop_reason,
                    usage=turn.usage.as_dict(),
                )
                history.append({"role": "assistant", "content": turn.content})

                if stop_reason == "refusal":
                    # 拒绝是终态：原样呈现给用户，不自动重试
                    outcome.ok, outcome.stop_reason = True, "refusal"
                    break
                if stop_reason == "model_context_window_exceeded":
                    await self._emit_error(
                        "context_window_exceeded",
                        "Conversation exceeded the model context window",
                    )
                    outcome.ok, outcome.stop_reason = False, stop_reason
                    outcome.error_code = "context_window_exceeded"
                    break
                if stop_reason in ("end_turn", "stop_sequence") or stop_reason is None:
                    outcome.ok, outcome.stop_reason = True, stop_reason or "end_turn"
                    break

                # ---- pause_turn：续传——消息已追加，直接进入下一轮 ----
                if stop_reason == "pause_turn":
                    continue

                # ---- tool_use：执行工具并回填结果，进入下一轮 ----
                if stop_reason == "tool_use":
                    await self._execute_tools(session_id, assistant_msg.id, turn.content, history)
                    continue

                # 未知 stop_reason：按终态收尾（防御）
                outcome.ok, outcome.stop_reason = True, str(stop_reason)
                break

            else:
                # ---- 全局 max_iterations 保护（：与熔断双保险）----
                await self._emit_error(
                    "max_iterations",
                    f"Agent loop hit max_iterations={self._max_iterations}; "
                    "stopping and reporting progress",
                )
                outcome.stop_reason = "max_iterations"
                outcome.error_code = "max_iterations"
        except asyncio.CancelledError:
            # ：中断 → finally 负责孤儿合成与状态回收，这里原样上抛
            raise
        finally:
            # 进度 ticker 先停：turn_complete 之前的尾部事件序列保持干净
            if self._progress_task is not None:
                self._progress_task.cancel()
                await asyncio.gather(self._progress_task, return_exceptions=True)
                self._progress_task = None
            await self._ensure_no_orphans(session_id, history)
            try:
                await self._repo.update_session_status(session_id, status)
            except Exception:
                pass

        outcome.usage = UsageData(
            input_tokens=ledger.total_input_tokens,
            output_tokens=ledger.total_output_tokens,
            cache_read=ledger.total_cache_read_tokens,
            cache_creation=ledger.total_cache_creation_tokens,
        )
        outcome.cost_usd = round(run_cost, 6)
        await self._emit_turn_complete(outcome, ledger)
        return outcome

    # ------------------------------------------------------------------
    # 任务拆解选择门：plan → task_options → 等待选择 → 注记
    # ------------------------------------------------------------------

    async def _plan_and_ask(
        self, session_id: str, user_text: str, history: list[dict[str, Any]]
    ) -> PlanResult | None:
        """规划器分析 → 需要时发 task_options 并等待用户选择。

        - simple / 规划器不可用 / 降级 → 直接返回（不打扰用户）
        - 选择超时（choice_waiter 返回 None）→ 自动采用 recommended 方案
        - 选定结果（含"自动采用"标记）以 user 注记消息**落库** + 进内存
          镜像 + 经 text_delta(role=user) 发实时流——三条路径一致，重启
          恢复 / 重连重放 / 在线客户端看到相同的拆解上下文
        - 返回 PlanResult 供 run() 读取 long_running 标记（None = 未规划）
        """
        if self._planner is None:
            return None
        try:
            plan: PlanResult = await self._planner.analyze(
                user_text, history[:-1] if history else []
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # 双保险：规划器自身已兜底，这里防实现疏漏
            logger.warning("planner.analyze escaped; skipping task planning", exc_info=True)
            return None
        if not plan.needs_choice:
            return plan

        options = plan.options[: self._planner.max_options] if hasattr(
            self._planner, "max_options"
        ) else plan.options
        choice_id = f"choice_{uuid4().hex[:12]}"
        await self._emit(
            SSEEventType.TASK_OPTIONS,
            TaskOptionsData(
                choice_id=choice_id,
                complexity=plan.complexity,
                options=[TaskOptionData(label=o.label, detail=o.detail) for o in options],
                recommended=min(plan.recommended, len(options) - 1),
                reason=plan.reason or None,
                timeout_s=self._choice_timeout_s,
            ),
        )

        if self._choice_waiter is None:
            # 无等待通道（裸 loop / 测试只发事件）→ 不阻塞，按推荐继续
            selected: int | None = None
        else:
            selected = await self._choice_waiter(choice_id)
        index = selected if (selected is not None and 0 <= selected < len(options)) else (
            min(plan.recommended, len(options) - 1)
        )
        option = options[index]
        source = "用户选择" if selected is not None else "未选择，自动采用推荐"
        label = COMPLEXITY_LABELS.get(plan.complexity, plan.complexity)
        note = f"【任务拆解】{label}任务 · {source}：{option.label}"
        if option.detail:
            note += f" —— {option.detail}"

        await self._repo.append_message(
            session_id, "user", [{"type": "text", "text": note}]
        )
        history.append({"role": "user", "content": [{"type": "text", "text": note}]})
        await self._emit(SSEEventType.TEXT_DELTA, TextDeltaData(text=note, role="user"))
        return plan

    # ------------------------------------------------------------------
    # 长时任务进度推送：后端扫描定义 + 分钟级 ticker
    # ------------------------------------------------------------------

    async def _progress_ticker(self, session_id: str) -> None:
        """长时任务进度 ticker（每 progress_interval_s 发一次 task_progress）。

        长时任务的"扫描定义"两级来源：
        1. 规划器标记 long_running → ticker 首个事件即按进度间隔发出；
        2. 未标记 → 先睡 long_running_after_s，run 仍在进行则自动升级为
           长时任务（防御规划器误判/降级漏标）。
        取消即静默退出（run 的 finally 统一回收）。
        """
        try:
            if self._long_running:
                await asyncio.sleep(self._progress_interval_s)
            else:
                await asyncio.sleep(self._long_running_after_s)
                self._long_running = True  # 运行时升级：仍在跑 → 视为长时
            while True:
                await self._emit_progress(session_id)
                await asyncio.sleep(self._progress_interval_s)
        except asyncio.CancelledError:
            raise

    async def _emit_progress(self, session_id: str) -> None:
        elapsed = int(time.monotonic() - self._run_started)
        await self._emit(
            SSEEventType.TASK_PROGRESS,
            TaskProgressData(
                elapsed_s=elapsed,
                iterations=self._iteration,
                last_activity=self._last_activity or None,
            ),
        )

    # ------------------------------------------------------------------
    # compaction：接近上下文阈值时压缩
    # ------------------------------------------------------------------

    async def _maybe_compact(
        self,
        session_id: str,
        ledger: TokenLedger,
        history: list[dict[str, Any]],
        model: str,
    ) -> bool:
        """执行一次 compaction；发生则返回 True。

        关键坑：必须把**完整 response.content**（含 compaction 块）
        回传进对话历史——追加为 assistant 消息（落库 + 内存镜像），只回传
        文本会静默丢失压缩状态。每 run 至多一次；失败降级不阻塞主循环
        （下轮 / 下次 run 的 usage 会再次触发）。
        """
        if self._compactor is None or self._compacted_once:
            return False
        self._compacted_once = True
        self._last_activity = "压缩对话上下文"
        try:
            content = await ledger.compact(
                self._compactor,
                messages=_merge_consecutive(history),
                system=self._system_blocks,
                tools=self._registry.api_schemas(),
                model=model,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("compaction failed; continuing without it: %r", exc)
            return False
        await self._repo.append_message(session_id, "assistant", content)
        history.append({"role": "assistant", "content": content})
        logger.info(
            "compaction applied for session %s (%d content blocks)",
            session_id,
            len(content),
        )
        return True

    # ------------------------------------------------------------------
    # 工具执行波次（/）
    # ------------------------------------------------------------------

    async def _execute_tools(
        self,
        session_id: str,
        assistant_message_id: str,
        content: Sequence[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> None:
        tool_blocks = [
            b for b in content
            if b.get("type") == "tool_use" and b.get("id") and b.get("name")
        ]
        if not tool_blocks:
            return

        for block in tool_blocks:
            await self._emit(
                SSEEventType.TOOL_CALL_START,
                ToolCallStartData(
                    tool_use_id=block["id"],
                    name=block["name"],
                    input_preview=json.dumps(
                        block.get("input", {}), ensure_ascii=False
                    )[:PREVIEW_LEN],
                ),
            )

        calls = [
            (block["id"], block["name"], dict(block.get("input") or {}))
            for block in tool_blocks
        ]
        names = [c[1] for c in calls]
        shown = "、".join(names[:3]) + ("…" if len(names) > 3 else "")
        self._last_activity = f"执行工具：{shown}"
        try:
            results = await self._dispatcher.dispatch_many(calls)
        except asyncio.CancelledError:
            # ：中断 → 全部未完成块合成 interrupted 结果，持久化，再上抛
            await self._persist_tool_results(
                session_id, assistant_message_id, tool_blocks, [],
                interrupted_pending=True, history=history, assistant_content=list(content),
            )
            raise

        self._last_activity = "整合工具结果"
        await self._persist_tool_results(
            session_id, assistant_message_id, tool_blocks, results,
            interrupted_pending=False, history=history, assistant_content=list(content),
        )

    async def _persist_tool_results(
        self,
        session_id: str,
        assistant_message_id: str,
        tool_blocks: Sequence[dict[str, Any]],
        results: Sequence[ToolResult | None],
        *,
        interrupted_pending: bool,
        history: list[dict[str, Any]],
        assistant_content: Sequence[dict[str, Any]],
    ) -> None:
        """tool_call 落审计表 + tool_result 落消息表 + SSE tool_call_result，
        并把 assistant / tool_result 两条消息同步进内存 history
        （中断路径也同步——否则 finally 的孤儿校验会误判补重复消息）。

        interrupted_pending=True 时（执行被取消），results 视为空，
        所有块合成 interrupted 结果。
        """
        result_blocks: list[dict[str, Any]] = []
        for index, block in enumerate(tool_blocks):
            if interrupted_pending or index >= len(results):
                tool_result_content = _interrupted_content(block)
                is_error, call_status, approval, duration_ms = (
                    True, "interrupted", "auto", None,
                )
            else:
                result = results[index]
                tool_result_content = result.content
                is_error, call_status = result.is_error, result.status
                approval, duration_ms = result.approval, result.duration_ms

            await self._repo.record_tool_call(
                session_id=session_id,
                message_id=assistant_message_id,
                tool_use_id=block["id"],
                tool_name=block["name"],
                tool_input=dict(block.get("input") or {}),
                result=tool_result_content,
                is_error=is_error,
                status=call_status,
                approval=approval,
                duration_ms=duration_ms,
            )
            await self._emit(
                SSEEventType.TOOL_CALL_RESULT,
                ToolCallResultData(
                    tool_use_id=block["id"],
                    is_error=is_error,
                    output_preview=tool_result_content[:PREVIEW_LEN],
                ),
            )
            result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": tool_result_content,
                "is_error": is_error,
            })

        await self._repo.append_message(session_id, "user", result_blocks)

        # 内存镜像与 DB 同步（append-only：只追加，不改既有项）
        history.append({"role": "assistant", "content": list(assistant_content)})
        history.append({"role": "user", "content": result_blocks})

    # ------------------------------------------------------------------
    # 不变量 1 防御校验：每轮结束前无孤儿 tool_use
    # ------------------------------------------------------------------

    async def _ensure_no_orphans(
        self, session_id: str, history: list[dict[str, Any]]
    ) -> None:
        if not needs_repair(history):
            return
        repaired = repair_orphans(history)
        for entry in repaired[len(history):]:
            # 正常路径到不了这里（tool_use 必配 tool_result）；到达即说明
            # 出现了意外路径的孤儿——当场合成 interrupted 结果并持久化
            await self._repo.append_message(session_id, entry["role"], entry["content"])
        history[:] = repaired

    # ------------------------------------------------------------------
    # SSE 事件发射（event_sink 由 API 层波次接 event_bus）
    # ------------------------------------------------------------------

    def _delta_emitter(self) -> Callable[[Any], Awaitable[None]]:
        async def on_delta(delta: Any) -> None:
            if delta.kind == "text":
                await self._emit(SSEEventType.TEXT_DELTA, TextDeltaData(text=delta.text))
            elif delta.kind == "thinking":
                await self._emit(
                    SSEEventType.THINKING_DELTA, ThinkingDeltaData(text=delta.text)
                )

        return on_delta

    async def _emit(self, event_type: SSEEventType, data: Any) -> None:
        if self._event_sink is None:
            return
        seq = self._event_seq + 1
        event = make_event(seq, event_type, data)
        self._event_seq = seq
        try:
            await self._event_sink(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 事件管道坏了不影响 loop（DB 是唯一事实来源）
            pass

    async def _emit_error(self, code: str, message: str) -> None:
        await self._emit(SSEEventType.ERROR, ErrorData(code=code, message=message))

    async def _emit_turn_complete(self, outcome: RunOutcome, ledger: TokenLedger) -> None:
        await self._emit(
            SSEEventType.TURN_COMPLETE,
            TurnCompleteData(
                stop_reason=outcome.stop_reason,
                usage=UsageOut(
                    input_tokens=ledger.total_input_tokens,
                    output_tokens=ledger.total_output_tokens,
                    cache_read=ledger.total_cache_read_tokens,
                    cache_creation=ledger.total_cache_creation_tokens,
                ),
                cost_usd=outcome.cost_usd,  # 定价表真实计算值（）
            ),
        )

    def _approval_notifier(self) -> Any:
        """把 dispatcher 的审批通知转成 approval_request SSE。"""

        async def notify(tool_name: str, tool_input: Any, tool_use_id: str) -> None:
            if self._event_sink is None:
                return
            seq = self._event_seq + 1
            event = make_event(
                seq,
                SSEEventType.APPROVAL_REQUEST,
                ApprovalRequestData(
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    input=dict(tool_input),
                    reason="dangerous tool requires user approval",
                ),
            )
            self._event_seq = seq
            try:
                await self._event_sink(event)
            except Exception:
                pass

        return notify


def _merge_consecutive(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并连续同角色消息（pause_turn 续传 / repair 插入后会出现）。

    Anthropic API 对消息序列要求 user/assistant 轮次交替；合并只拼接
    content 列表，不修改任何既有块（append-only 友好）。
    """
    merged: list[dict[str, Any]] = []
    for message in history:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {
                "role": message["role"],
                "content": list(merged[-1]["content"]) + list(message["content"]),
            }
        else:
            merged.append({"role": message["role"], "content": list(message["content"])})
    return merged


def _interrupted_content(block: dict[str, Any]) -> str:
    """给未执行完的 tool_use 合成 interrupted 结果文本（/）。"""
    synthetic = synthetic_tool_result(block["id"])
    return str(synthetic["content"])
