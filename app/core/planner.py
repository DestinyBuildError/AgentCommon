"""任务拆解规划器（后端思考能力增强）。

在消息进入 agent loop 之前做一次轻量 LLM 分析：
- 判定任务复杂度（simple / medium / complex）
- medium / complex → 生成 2~4 个有实质差异的执行方案选项，
  经 task_options SSE 事件交给用户选择后再开始执行
- simple → 直接放行，不打扰用户

设计约束（与 一致）：
- **永不阻塞主链路**：analyze() 内部吞掉一切异常，任何失败（网络 / 解析 /
  字段非法）都降级为"简单任务直接执行"——规划器是增强能力，不是正确性依赖
- 单独一次非流式 create 调用，不用 agent loop 的 run_turn 流式管线——
  规划不需要 thinking / tools / deltas
- **勿传 temperature**：anthropic 2.x 的 create() 已移除该直传参数
  （2.6.0 实测 TypeError: unexpected keyword argument），规划确定性由
  提示词的"只输出一个 JSON 对象"硬约束保证
- 输出严格 JSON，解析层做防御：剥 code fence、截取首尾大括号、字段
  逐项校验，任何不合法直接降级
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

COMPLEXITY_LABELS = {"simple": "简单", "medium": "中等", "complex": "复杂"}
_VALID_COMPLEXITY = frozenset(COMPLEXITY_LABELS)
_MIN_OPTIONS = 2  # 少于 2 个选项视为"不需要选择"（单选项没有选择意义）

_PLANNER_SYSTEM = """你是聊天 Agent 的任务规划器。分析用户的最新消息，判断任务复杂度与意图清晰度，并决定是否需要先让用户确认执行方向，再开始干活。

【模糊度优先】动手前先判断意图是否清晰。以下情况属于"模糊不清"，即使任务量不大也要拆出选项让用户确认，宁可多问一句也不要猜错方向白干：
- 指代不明："改一下这个文件""处理那批数据"——哪个文件？哪批数据？
- 缺关键参数：导出（什么格式？）、生成报告（什么主题/篇幅？）、写个脚本（用什么语言/实现什么？）
- 存在多种合理解读：同一句话可以理解为完全不同的任务

复杂度判定：
- simple：单一明确动作——问候/闲聊、翻译一段话、解释概念、有明确指代的追问、对上文选择的回应（如"就用方案2"）→ options 为 []
- medium：任务存在多种合理做法或范围不明确（如"写个爬虫""做个网站"）→ 给出 2~4 个具体执行方案
- complex：大型任务（完整应用、跨文件重构、多数据源项目）→ 给出 2~4 个由小到大的实施范围选项（最小可用版 / 标准版 / 完整版等）

长时任务判定（long_running）：任务会持续较长时间——批量处理、多文件改动、爬取数据、生成长文/大段代码、多步骤工程 → true；一句话问答、小改动 → false。

输出要求：只输出一个 JSON 对象，禁止任何其他文字、解释或代码块标记：
{"complexity": "simple|medium|complex", "reason": "一句话理由",
 "options": [{"label": "方案名(不超过12字)", "detail": "做法/范围说明(不超过40字)"}],
 "recommended": 0, "long_running": false, "progress_label": "任务简述(不超过20字)"}

- options 为空列表时 recommended 填 0
- recommended 必须是 options 的合法下标
- 选项之间必须有实质差异（不同做法 / 不同范围 / 不同技术路线），不是同义改写
- 用户消息里已经说清方向时不要硬拆：方向明确就直接执行
- progress_label 给出任务的一句话简述，长时任务时前端用于进度展示；简短任务给空串"""


@dataclass(frozen=True)
class PlanOption:
    """一个执行方案（label 给卡片标题，detail 给副文本）。"""

    label: str
    detail: str = ""


@dataclass(frozen=True)
class PlanResult:
    """规划器分析结果（AgentLoop 消费）。

    needs_choice 为 True 时 AgentLoop 发 task_options 事件并挂起等待用户；
    long_running 为 True 时启动分钟级进度推送（task_progress 事件）。
    """

    complexity: str = "simple"
    reason: str = ""
    options: tuple[PlanOption, ...] = ()
    recommended: int = 0
    long_running: bool = False     # 长时任务 → 前端展示分钟级进度
    progress_label: str = ""       # 任务一句话简述（进度卡展示用）

    @property
    def needs_choice(self) -> bool:
        return len(self.options) >= _MIN_OPTIONS

    @property
    def complexity_label(self) -> str:
        return COMPLEXITY_LABELS.get(self.complexity, self.complexity)


def parse_plan(raw: str, *, max_options: int = 4) -> PlanResult:
    """把规划器输出文本解析为 PlanResult；任何不合法 → 降级 simple。

    防御链：剥 ``` fence → 截取首 '{' 到末 '}' → json.loads → 字段逐项
    校验（complexity 白名单 / options 非空 dict 且 label 非空 / recommended
    夹取到合法下标）。
    """
    result = PlanResult()
    if not raw or not raw.strip():
        return result
    text = raw.strip()
    if text.startswith("```"):
        # 剥首行 ```（含语言标记）与末行 ```
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().endswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return result
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return result
    if not isinstance(data, dict):
        return result

    complexity = str(data.get("complexity") or "").strip().lower()
    if complexity not in _VALID_COMPLEXITY:
        return result
    reason = str(data.get("reason") or "").strip()

    options: list[PlanOption] = []
    raw_options = data.get("options")
    if isinstance(raw_options, list):
        for item in raw_options[:max_options]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            options.append(PlanOption(label=label, detail=str(item.get("detail") or "").strip()))
    if complexity != "simple" and len(options) < _MIN_OPTIONS:
        # 声称要拆但给不出有效选项 → 整体降级 simple 直接执行（选项残缺比
        # 打扰更糟；保留 medium 标签却无可选项会让事件/注记语义不完整）
        return PlanResult()

    recommended = data.get("recommended")
    idx = recommended if isinstance(recommended, int) else 0
    if options:
        idx = min(max(idx, 0), len(options) - 1)
    else:
        idx = 0
    progress_label = str(data.get("progress_label") or "").strip()[:40]
    return PlanResult(
        complexity=complexity,
        reason=reason,
        options=tuple(options),
        recommended=idx,
        long_running=bool(data.get("long_running")),
        progress_label=progress_label,
    )


def _history_tail_text(history: list[dict[str, Any]], *, max_messages: int = 8) -> str:
    """把近期历史压成纯文本便于规划：只取 text 块、逐条截断、最多 N 条。"""
    lines: list[str] = []
    for message in history[-max_messages:]:
        role = str(message.get("role") or "?")
        chunks: list[str] = []
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text") or ""))
        text = " ".join(chunks).strip()
        if text:
            lines.append(f"{role}: {text[:160]}")
    return "\n".join(lines)


class TaskPlanner:
    """LLM 任务分级器。注入 client 即可 mock（与 LLMClient 同模式）。"""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_options: int = 4,
        max_tokens: int = 700,
        history_tail_chars: int = 1200,
    ) -> None:
        self._client = client
        self._model = model
        self._max_options = max(2, int(max_options))
        self._max_tokens = int(max_tokens)
        self._history_tail_chars = int(history_tail_chars)

    async def analyze(
        self, user_text: str, history: list[dict[str, Any]] | None = None
    ) -> PlanResult:
        """分析用户消息。**永不抛异常**——任何失败返回 simple 直接执行。"""
        try:
            return await self._analyze_inner(user_text, history or [])
        except Exception:
            logger.warning("task planner failed; falling back to simple", exc_info=True)
            return PlanResult()

    async def _analyze_inner(
        self, user_text: str, history: list[dict[str, Any]]
    ) -> PlanResult:
        tail = _history_tail_text(history)[: self._history_tail_chars]
        content = ""
        if tail:
            content += f"<近期对话>\n{tail}\n</近期对话>\n\n"
        content += f"<用户最新消息>\n{user_text[:2000]}\n</用户最新消息>"
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_PLANNER_SYSTEM,
            messages=[{"role": "user", "content": [{"type": "text", "text": content}]}],
        )
        raw = "".join(
            str(block.text)
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        )
        plan = parse_plan(raw, max_options=self._max_options)
        logger.info(
            "task planner: complexity=%s options=%d (raw %d chars)",
            plan.complexity,
            len(plan.options),
            len(raw),
        )
        return plan
