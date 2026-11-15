"""任务拆解规划器测试—— 全部 mock LLM，绝不发起真实 API 调用。

覆盖三层：
- parse_plan：JSON 解析防御链（fence 剥离 / 大括号截取 / 字段校验 / 降级）
- TaskPlanner.analyze：client 异常 / 空文本 → 降级 simple，永不抛异常
- AgentLoop 集成：simple 不打扰；medium 发 task_options → 用户选择注入
  注记；等待器返回 None（超时语义）→ 自动采用推荐方案
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.agent_loop import AgentLoop
from app.core.dispatcher import ToolDispatcher
from app.core.llm_client import LLMTurnResult, UsageData
from app.core.planner import PlanOption, PlanResult, TaskPlanner, parse_plan
from app.db.database import build_engine
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import SSEEventType
from app.tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# 夹具（与 test_agent_loop 同模式：内存库 + 最小 registry）
# ----------------------------------------------------------------------


@pytest.fixture
async def db_session() -> AsyncSession:
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


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()  # 规划器测试不执行工具，空 registry 即可


class ScriptedLLM:
    """按序回放 LLMTurnResult（与 test_agent_loop 同名同语义）。"""

    def __init__(self, turns: list[Any], delay_s: float = 0.0) -> None:
        self.turns = list(turns)
        self.delay_s = delay_s  # 每次 run_turn 前模拟耗时（进度 ticker 测试用）
        self.calls: list[dict[str, Any]] = []

    async def run_turn(self, *, messages, **kwargs) -> LLMTurnResult:
        self.calls.append({"messages": messages})
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def end_turn(text: str = "done") -> LLMTurnResult:
    return LLMTurnResult(
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=UsageData(10, 5, 8),
    )


class FakePlanner:
    """脚本化规划器：analyze 按序回放 PlanResult，并记录调用入参。"""

    def __init__(self, results: list[PlanResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.max_options = 4

    async def analyze(self, user_text: str, history: list[dict[str, Any]]) -> PlanResult:
        self.calls.append((user_text, history))
        return self.results.pop(0)


def build_loop(
    repo: Repository,
    registry: ToolRegistry,
    llm: ScriptedLLM,
    *,
    planner: Any | None = None,
    choice_waiter: Any | None = None,
    event_sink=None,
    progress_interval_s: float = 60.0,
    long_running_after_s: float = 90.0,
) -> AgentLoop:
    dispatcher = ToolDispatcher(registry, retry_backoff_base_s=0)
    return AgentLoop(
        llm=llm,
        repo=repo,
        registry=registry,
        dispatcher=dispatcher,
        system_prompt="You are a test agent.",
        event_sink=event_sink,
        model="claude-opus-4-8",
        planner=planner,
        choice_waiter=choice_waiter,
        choice_timeout_s=120,
        progress_interval_s=progress_interval_s,
        long_running_after_s=long_running_after_s,
    )


PLAN_MEDIUM = PlanResult(
    complexity="medium",
    reason="多种做法",
    options=(PlanOption(label="最小脚本", detail="单文件"), PlanOption(label="完整项目", detail="含测试")),
    recommended=1,
)


# ----------------------------------------------------------------------
# parse_plan：解析防御链
# ----------------------------------------------------------------------


def test_parse_plan_valid_json() -> None:
    raw = json.dumps(
        {
            "complexity": "medium",
            "reason": "范围不明",
            "options": [
                {"label": "方案A", "detail": "做法一"},
                {"label": "方案B", "detail": ""},
            ],
            "recommended": 1,
        },
        ensure_ascii=False,
    )
    plan = parse_plan(raw)
    assert plan.complexity == "medium"
    assert plan.reason == "范围不明"
    assert len(plan.options) == 2
    assert plan.options[1].label == "方案B"
    assert plan.recommended == 1
    assert plan.needs_choice


def test_parse_plan_strips_code_fence_and_noise() -> None:
    inner = '{"complexity": "complex", "reason": "r", "options": [{"label": "a"}, {"label": "b"}], "recommended": 0}'
    for raw in (f"```json\n{inner}\n```", f"好的，分析如下：{inner} 以上。"):
        plan = parse_plan(raw)
        assert plan.complexity == "complex"
        assert plan.needs_choice


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "不是 JSON",
        '{"complexity": "hard", "options": []}',  # 复杂度白名单外
        '{"options": [{"label": "a"}, {"label": "b"}]}',  # 缺 complexity
        '{"complexity": "medium", "options": "no"}',  # options 非列表
        '{"complexity": "medium", "options": [{"detail": "没 label"}]}',  # 全部无效选项
    ],
)
def test_parse_plan_degrades_to_simple(raw: str) -> None:
    plan = parse_plan(raw)
    assert plan.complexity == "simple"
    assert not plan.needs_choice


def test_parse_plan_medium_claim_with_one_option_degrades() -> None:
    # 声称 medium 但有效选项不足 2 → 降级（残缺选项比打扰更糟）
    raw = '{"complexity": "medium", "options": [{"label": "唯一"}], "recommended": 0}'
    plan = parse_plan(raw)
    assert not plan.needs_choice


def test_parse_plan_recommended_clamped() -> None:
    raw = '{"complexity": "medium", "options": [{"label": "a"}, {"label": "b"}], "recommended": 99}'
    assert parse_plan(raw).recommended == 1


def test_parse_plan_long_running_flag() -> None:
    raw = (
        '{"complexity": "simple", "reason": "r", "options": [], "recommended": 0,'
        ' "long_running": true, "progress_label": "批量转换文件"}'
    )
    plan = parse_plan(raw)
    assert plan.long_running
    assert plan.progress_label == "批量转换文件"
    assert not plan.needs_choice  # 简单但长时：直接跑，进度卡兜底


def test_parse_plan_degraded_not_long_running() -> None:
    # 一切降级路径（坏 JSON / 白名单外复杂度）long_running 必须为 False
    for raw in ("垃圾", '{"complexity": "hard"}', '{"complexity": "simple"}'):
        assert not parse_plan(raw).long_running


def test_parse_plan_max_options_cap() -> None:
    options = [{"label": f"o{i}"} for i in range(6)]
    raw = json.dumps({"complexity": "complex", "options": options, "recommended": 0})
    assert len(parse_plan(raw, max_options=4).options) == 4


# ----------------------------------------------------------------------
# TaskPlanner.analyze：降级保证（永不抛异常）
# ----------------------------------------------------------------------


class FakePlannerClient:
    """client.messages.create 的脚本化替身（返回 text 块或抛异常）。"""

    def __init__(self, texts: list[str | None] | None = None, exc: Exception | None = None) -> None:
        self.texts = list(texts or [])
        self.exc = exc
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # TaskPlanner 经 client.messages.create 调用

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        text = self.texts.pop(0)
        blocks = []
        if text is not None:
            blocks.append(SimpleNamespace(type="text", text=text))
        return SimpleNamespace(content=blocks)


@pytest.mark.asyncio
async def test_planner_analyze_parses_json() -> None:
    client = FakePlannerClient(
        texts=[
            json.dumps(
                {
                    "complexity": "medium",
                    "reason": "多做法",
                    "options": [{"label": "A"}, {"label": "B"}],
                    "recommended": 0,
                }
            )
        ]
    )
    planner = TaskPlanner(client, model="m")
    plan = await planner.analyze("写个爬虫", [])
    assert plan.needs_choice
    assert plan.options[0].label == "A"
    # 规划请求走 create（非流式），system 为规划器指令
    assert client.calls and "规划器" in client.calls[0]["system"]


@pytest.mark.asyncio
async def test_planner_analyze_client_error_degrades() -> None:
    planner = TaskPlanner(FakePlannerClient(exc=RuntimeError("network down")), model="m")
    plan = await planner.analyze("做网站", [])
    assert plan == PlanResult()


@pytest.mark.asyncio
async def test_planner_analyze_no_text_blocks_degrades() -> None:
    planner = TaskPlanner(FakePlannerClient(texts=[None]), model="m")
    assert not (await planner.analyze("x", [])).needs_choice


@pytest.mark.asyncio
async def test_planner_history_tail_included() -> None:
    client = FakePlannerClient(texts=["{}"])
    planner = TaskPlanner(client, model="m")
    history = [{"role": "user", "content": [{"type": "text", "text": "之前聊过的内容"}]}]
    await planner.analyze("继续", history)
    sent = client.calls[0]["messages"][0]["content"][0]["text"]
    assert "之前聊过的内容" in sent


# ----------------------------------------------------------------------
# AgentLoop 集成：选择门全链路
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_task_no_task_options(repo: Repository, registry: ToolRegistry) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()])
    planner = FakePlanner([PlanResult()])  # simple → 不打扰
    events: list[Any] = []
    loop = build_loop(repo, registry, llm, planner=planner, event_sink=events.append)
    outcome = await loop.run(session.id, "你好")
    assert outcome.ok
    assert all(e.type != SSEEventType.TASK_OPTIONS for e in events)
    # 规划器拿到的 history 不含刚追加的当前消息
    assert planner.calls[0][0] == "你好"


@pytest.mark.asyncio
async def test_medium_task_user_choice_injected(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn("按方案干完了")])
    planner = FakePlanner([PLAN_MEDIUM])
    events: list[Any] = []
    captured: dict[str, int] = {}

    async def waiter(choice_id: str) -> int | None:
        captured["choice_id"] = -1  # 占位；真实 id 由事件侧校验
        return 1  # 用户选择方案 2

    loop = build_loop(repo, registry, llm, planner=planner, choice_waiter=waiter, event_sink=events.append)
    outcome = await loop.run(session.id, "做个数据看板")
    assert outcome.ok

    # ① task_options 事件载荷正确
    task_events = [e for e in events if e.type == SSEEventType.TASK_OPTIONS]
    assert len(task_events) == 1
    payload = task_events[0].data
    assert payload["complexity"] == "medium"
    assert payload["recommended"] == 1
    assert payload["options"][1]["label"] == "完整项目"
    assert payload["choice_id"].startswith("choice_")

    # ② 选择注记以 user 消息落库 + 实时流带 role=user
    note_events = [
        e for e in events if e.type == SSEEventType.TEXT_DELTA and e.data.get("role") == "user"
    ]
    assert note_events and "用户选择" in note_events[0].data["text"]
    assert "完整项目" in note_events[0].data["text"]
    messages = await repo.get_messages(session.id)
    assert any("【任务拆解】" in str(m.content) for m in messages if m.role == "user")

    # ③ 主 LLM 看到的 history 含选择注记（模型按选定方向执行）
    llm_messages = llm.calls[0]["messages"]
    assert any("完整项目" in str(b.get("text", "")) for m in llm_messages for b in m["content"])


@pytest.mark.asyncio
async def test_choice_timeout_auto_adopts_recommended(
    repo: Repository, registry: ToolRegistry
) -> None:
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()])
    planner = FakePlanner([PLAN_MEDIUM])  # recommended=1
    events: list[Any] = []

    async def waiter(choice_id: str) -> int | None:
        return None  # 超时语义（SessionManager.wait_choice 超时时返回 None）

    loop = build_loop(repo, registry, llm, planner=planner, choice_waiter=waiter, event_sink=events.append)
    outcome = await loop.run(session.id, "做个数据看板")
    assert outcome.ok
    note_events = [
        e for e in events if e.type == SSEEventType.TEXT_DELTA and e.data.get("role") == "user"
    ]
    assert note_events and "自动采用推荐" in note_events[0].data["text"]
    assert "完整项目" in note_events[0].data["text"]  # 采用的是 recommended=1


@pytest.mark.asyncio
async def test_choice_resolved_async_after_emit(
    repo: Repository, registry: ToolRegistry
) -> None:
    """选择门真实时序：task_options 发出后 waiter 挂起，事件侧稍后 resolve。"""
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()])
    planner = FakePlanner([PLAN_MEDIUM])
    events: list[Any] = []
    futures: dict[str, asyncio.Future] = {}

    async def waiter(choice_id: str) -> int | None:
        fut = asyncio.get_running_loop().create_future()
        futures[choice_id] = fut
        return await fut  # 挂起直到测试侧 set_result

    async def resolve_when_emitted() -> None:
        while not futures:
            await asyncio.sleep(0)
        choice_id = next(iter(futures))
        # 模拟 POST /choices：选方案 0
        task_events = [e for e in events if e.type == SSEEventType.TASK_OPTIONS]
        assert task_events and task_events[0].data["choice_id"] == choice_id
        futures[choice_id].set_result(0)

    loop = build_loop(repo, registry, llm, planner=planner, choice_waiter=waiter, event_sink=events.append)
    resolver = asyncio.ensure_future(resolve_when_emitted())
    outcome = await loop.run(session.id, "做个数据看板")
    await resolver
    assert outcome.ok
    note_events = [
        e for e in events if e.type == SSEEventType.TEXT_DELTA and e.data.get("role") == "user"
    ]
    assert note_events and "最小脚本" in note_events[0].data["text"]


@pytest.mark.asyncio
async def test_planner_none_skips_gate(repo: Repository, registry: ToolRegistry) -> None:
    """planner=None（禁用/降级开关）→ 行为与改造前完全一致。"""
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()])
    events: list[Any] = []
    loop = build_loop(repo, registry, llm, event_sink=events.append)
    outcome = await loop.run(session.id, "hello")
    assert outcome.ok
    assert all(e.type != SSEEventType.TASK_OPTIONS for e in events)


# ----------------------------------------------------------------------
# 长时任务进度推送：扫描定义 + 分钟级 ticker
# ----------------------------------------------------------------------


def progress_events(events: list[Any]) -> list[Any]:
    return [e for e in events if e.type == SSEEventType.TASK_PROGRESS]


@pytest.mark.asyncio
async def test_progress_emitted_when_planner_marks_long_running(
    repo: Repository, registry: ToolRegistry
) -> None:
    """规划器标记 long_running → 按间隔推送 task_progress。"""
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()], delay_s=1.5)  # 单轮耗时 1.5s，跨过 1s 间隔
    planner = FakePlanner(
        [PlanResult(long_running=True, progress_label="批量转换")]
    )
    events: list[Any] = []
    loop = build_loop(
        repo, registry, llm,
        planner=planner, event_sink=events.append,
        progress_interval_s=1.0,
    )
    outcome = await loop.run(session.id, "批量处理")
    assert outcome.ok
    ticks = progress_events(events)
    assert len(ticks) >= 1
    payload = ticks[0].data
    assert payload["elapsed_s"] >= 1
    assert payload["iterations"] >= 1
    assert payload["last_activity"]


@pytest.mark.asyncio
async def test_progress_runtime_upgrade_without_planner(
    repo: Repository, registry: ToolRegistry
) -> None:
    """规划器未标记（或降级）→ 运行超 long_running_after_s 自动升级为长时。"""
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()], delay_s=0.8)
    events: list[Any] = []
    loop = build_loop(
        repo, registry, llm, event_sink=events.append,
        progress_interval_s=0.4, long_running_after_s=0.3,
    )
    outcome = await loop.run(session.id, "慢慢跑")
    assert outcome.ok
    ticks = progress_events(events)
    assert len(ticks) >= 1  # 0.3s 升级 → 0.3/0.7(1.1) 推送，run 总长 ~0.8s+


@pytest.mark.asyncio
async def test_no_progress_for_short_task(
    repo: Repository, registry: ToolRegistry
) -> None:
    """短任务（run 先于升级阈值结束）→ 零 task_progress 事件。"""
    session = await repo.create_session(model="m", effort="high")
    llm = ScriptedLLM([end_turn()])
    events: list[Any] = []
    loop = build_loop(
        repo, registry, llm, event_sink=events.append,
        progress_interval_s=0.5, long_running_after_s=5.0,
    )
    outcome = await loop.run(session.id, "你好")
    assert outcome.ok
    assert not progress_events(events)
