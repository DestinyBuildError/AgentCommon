
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/dev_mock.db")
os.environ.setdefault("WORKSPACE_ROOT", "../workspace")

from app.core.config import get_settings
from app.core.dispatcher import ToolDispatcher
from app.db.database import get_sessionmaker
from fastapi import Request
from app.core.llm_client import LLMTurnResult, StreamDelta, UsageData
from app.core.session_manager import SessionManager
from app.tools.builtin import register_default_tools
from app.tools.registry import ToolRegistry, ToolSpec

WORKSPACE_ROOT = Path("../workspace/")

DELTA_CHUNK = 8


@dataclass
class MockTurn:

    text: str = ""
    thinking: str | None = None
    tool_use: dict[str, Any] | None = None
    stop_reason: str = "end_turn"
    usage: UsageData = field(
        default_factory=lambda: UsageData(input_tokens=120, output_tokens=30, cache_read=100)
    )
    delta_delay: float | None = None
    branch: Any = None


def _fallback_turn() -> MockTurn:
    return MockTurn(text="(mock) 继续。", usage=UsageData(80, 10, 70))


def scenario_chat() -> list[MockTurn]:
    return [
        MockTurn(
            text="你好！我是 mock Agent。当前运行在**无 API Key 的脚本模式**下，"
            "可以试试对我说「读一下文件」或「帮我部署」来演示工具调用与审批流。",
            thinking="用户在打招呼，介绍 mock 模式可用的演示场景。",
            usage=UsageData(input_tokens=120, output_tokens=48, cache_read=100),
        )
    ]


def scenario_file() -> list[MockTurn]:
    return [
        MockTurn(
            text="好的，我先读取工作区里的 notes.txt 看看内容。",
            tool_use={"name": "read_file", "input": {"path": "notes.txt"}},
            stop_reason="tool_use",
            usage=UsageData(input_tokens=140, output_tokens=24, cache_read=120),
        ),
        MockTurn(
            text="已经读完 notes.txt：这是 mock 工作区的示例文件，内容由 dev_mock_server "
            "在启动时写入。工具链路（tool_call_start → tool_call_result → 回答）走通了。",
            usage=UsageData(input_tokens=210, output_tokens=52, cache_read=180),
        ),
    ]


def scenario_approval() -> list[MockTurn]:
    return [
        MockTurn(
            text="部署属于危险操作，我需要先向你申请审批。",
            tool_use={
                "name": "deploy_tool",
                "input": {"env": "production", "version": "v1.2.3"},
            },
            stop_reason="tool_use",
            usage=UsageData(input_tokens=150, output_tokens=20, cache_read=130),
        ),
        MockTurn(text=""),
    ]


def scenario_slow() -> list[MockTurn]:
    body = "这是一条慢速长输出，用于测试断线重连：前端掉线后应指数退避重连，" \
           "并用 after_seq 续传，不丢事件、不重复渲染。"
    turn = MockTurn(
        text=body,
        usage=UsageData(input_tokens=90, output_tokens=64, cache_read=80),
        delta_delay=0.25,
    )
    return [turn]


def scenario_error() -> list[MockTurn]:
    return [RuntimeError("(mock) simulated LLM outage")]


def _ensure_workspace_demo_dir() -> Path:
    demo_dir = (WORKSPACE_ROOT / "extra_dir").resolve()
    demo_dir.mkdir(parents=True, exist_ok=True)
    note = demo_dir / "extra.txt"
    if not note.exists():
        note.write_text(
            "extra.txt — 多目录工作区演示文件\n"
            "由 dev_mock_server workspace 场景经 add_workspace_dir 加入后可读。\n",
            encoding="utf-8",
        )
    return demo_dir


def scenario_workspace() -> list[MockTurn]:
    demo_dir = _ensure_workspace_demo_dir()
    return [
        MockTurn(
            text="好的，我先把这个演示目录加入多目录工作区。",
            tool_use={
                "name": "add_workspace_dir",
                "input": {"path": str(demo_dir)},
            },
            stop_reason="tool_use",
            usage=UsageData(input_tokens=150, output_tokens=22, cache_read=130),
        ),
        MockTurn(
            text="目录已就绪，读取其中的 extra.txt 验证跨目录访问。",
            tool_use={
                "name": "read_file",
                "input": {"path": str(demo_dir / "extra.txt")},
            },
            stop_reason="tool_use",
            usage=UsageData(input_tokens=210, output_tokens=26, cache_read=180),
        ),
        MockTurn(
            text="已经读完 extra.txt（演示文件）。当前工作区目录集合如下：",
            tool_use={"name": "list_workspace_dirs", "input": {}},
            stop_reason="tool_use",
            usage=UsageData(input_tokens=240, output_tokens=24, cache_read=200),
        ),
        MockTurn(
            text="多目录工作区全链路走通：add_workspace_dir 落库并广播 "
            "workspace_changed → read_file 立即可读新目录 → list_workspace_dirs "
            "返回完整列表。工具执行期间实时查询目录集合，无需重注册/重启。",
            usage=UsageData(input_tokens=280, output_tokens=56, cache_read=230),
        ),
    ]


def scenario_tools() -> list[MockTurn]:
    return [
        MockTurn(
            text="我先在精选工具目录里搜一下有没有能做文本反转的工具。",
            tool_use={"name": "discover_tools", "input": {"query": "reverse text"}},
            stop_reason="tool_use",
            usage=UsageData(input_tokens=150, output_tokens=26, cache_read=130),
        ),
        MockTurn(
            text="目录里有 text_tools（builtin 信任级，含 reverse_text 等工具）。"
            "安装属于危险操作，我向你申请审批。",
            tool_use={"name": "install_mcp_server", "input": {"name": "text_tools"}},
            stop_reason="tool_use",
            usage=UsageData(input_tokens=210, output_tokens=34, cache_read=180),
        ),
        MockTurn(text="", branch=resolve_tools_install_branch),
        MockTurn(
            text="自扩展闭环走通：discover_tools 检索 → install_mcp_server 过审批门"
            "（成功后后端向全部会话广播 tools_changed，前端会 toast「工具集已更新」）"
            "→ 新工具在**同一 run** 的下一迭代立即可调；若安装被拒则自动降级 "
            "run_python 沙箱兜底。",
            usage=UsageData(input_tokens=300, output_tokens=64, cache_read=260),
        ),
    ]


def resolve_tools_install_branch(denied: bool) -> MockTurn:
    if denied:
        return MockTurn(
            text="收到审批结果：**未获批准**。没关系——run_python 沙箱是万能兜底，"
            "我直接写代码完成文本反转。",
            tool_use={
                "name": "run_python",
                "input": {"code": "print('abcdef'[::-1])"},
            },
            stop_reason="tool_use",
            usage=UsageData(input_tokens=250, output_tokens=38, cache_read=220),
        )
    return MockTurn(
        text="安装成功，text_tools 的 3 个工具已注册进注册表。现在直接调用 "
        "mcp__text_tools__reverse_text 反转 'abcdef'。",
        tool_use={
            "name": "mcp__text_tools__reverse_text",
            "input": {"text": "abcdef"},
        },
        stop_reason="tool_use",
        usage=UsageData(input_tokens=260, output_tokens=40, cache_read=230),
    )


SCENARIOS: dict[str, callable] = {
    "chat": scenario_chat,
    "file": scenario_file,
    "approval": scenario_approval,
    "slow": scenario_slow,
    "error": scenario_error,
    "workspace": scenario_workspace,
    "tools": scenario_tools,
}

_KEYWORD_ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("审批", "部署", "deploy", "approval", "危险"), "approval"),
    (("工具目录", "发现工具", "装工具", "discover", "动态工具"), "tools"),
    (("工作区", "workspace", "多目录"), "workspace"),
    (("文件", "读一下", "read", "file"), "file"),
    (("slow", "慢"), "slow"),
    (("error", "报错"), "error"),
]


def route_scenario(last_user_text: str) -> str:
    forced = os.environ.get("MOCK_SCENARIO")
    if forced and forced in SCENARIOS:
        return forced
    lowered = last_user_text.lower()
    for keywords, scenario in _KEYWORD_ROUTES:
        if any(k in lowered or k in last_user_text for k in keywords):
            return scenario
    return "chat"


class FakeLLM:

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.calls.append({k: kwargs[k] for k in kwargs if k != "on_delta"})
        on_delta = kwargs.get("on_delta")
        messages = kwargs.get("messages") or []

        turns = SCENARIOS[self._scenario_of(messages)]()
        index = self._iteration_index(messages)
        turn = turns[index] if index < len(turns) else _fallback_turn()
        if isinstance(turn, BaseException):
            raise turn
        if turn.text == "" and turn.tool_use is None:
            if turn.branch is not None:
                denied = self._last_tool_is_error(self._current_run_messages(messages))
                turn = turn.branch(denied)
            else:
                turn = self._resolve_branch_turn(turn, messages)

        if on_delta is not None:
            for delta in self._deltas_of(turn):
                if turn.delta_delay is not None:
                    await asyncio.sleep(turn.delta_delay)
                elif self._global_delay():
                    await asyncio.sleep(self._global_delay())
                await on_delta(delta)

        return self._result_of(turn)


    @staticmethod
    def _global_delay() -> float:
        try:
            return float(os.environ.get("MOCK_DELTA_DELAY", "0") or 0)
        except ValueError:
            return 0.0

    @staticmethod
    def _scenario_of(messages: list) -> str:
        return route_scenario(_last_user_text(messages))

    @staticmethod
    def _current_run_messages(messages: list) -> list:
        last_text_user = -1
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue
            if any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in (m.get("content") or [])
            ):
                last_text_user = i
        return messages[last_text_user + 1 :] if last_text_user >= 0 else messages

    def _iteration_index(self, messages: list) -> int:
        run_msgs = self._current_run_messages(messages)
        return sum(
            1
            for m in run_msgs
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        )

    @staticmethod
    def _last_tool_is_error(messages: list) -> bool:
        for m in reversed(messages):
            for b in reversed(m.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    return bool(b.get("is_error", False))
        return False

    def _resolve_branch_turn(self, turn: MockTurn, messages: list) -> MockTurn:
        denied = self._last_tool_is_error(self._current_run_messages(messages))
        if denied:
            return MockTurn(
                text="收到审批结果：**未获批准**。我不会重试该操作，"
                "建议改用灰度发布或手动执行，需要的话我可以生成操作手册。",
                usage=UsageData(input_tokens=230, output_tokens=44, cache_read=200),
                delta_delay=turn.delta_delay,
            )
        return MockTurn(
            text="审批已通过，deploy_tool 执行成功：production 环境已发布 v1.2.3。"
            "整条链路（approval_request → POST /approvals → 工具执行 → 收尾）走通了。",
            usage=UsageData(input_tokens=230, output_tokens=48, cache_read=200),
            delta_delay=turn.delta_delay,
        )

    @staticmethod
    def _deltas_of(turn: MockTurn) -> list[StreamDelta]:
        deltas: list[StreamDelta] = []
        if turn.thinking:
            deltas.append(StreamDelta(kind="thinking", text=turn.thinking))
        text = turn.text
        for i in range(0, len(text), DELTA_CHUNK):
            deltas.append(StreamDelta(kind="text", text=text[i : i + DELTA_CHUNK]))
        return deltas

    @staticmethod
    def _result_of(turn: MockTurn) -> LLMTurnResult:
        content: list[dict[str, Any]] = []
        if turn.thinking:
            content.append({"type": "thinking", "thinking": turn.thinking, "signature": "mock"})
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        if turn.tool_use is not None:
            content.append(
                {
                    "type": "tool_use",
                    "id": f"tu_{uuid.uuid4().hex[:8]}",
                    "name": turn.tool_use["name"],
                    "input": dict(turn.tool_use.get("input") or {}),
                }
            )
        return LLMTurnResult(
            content=content,
            stop_reason=turn.stop_reason if turn.tool_use else "end_turn",
            usage=turn.usage,
        )


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        parts = [
            str(b.get("text") or "")
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        if text:
            return text
    return ""


def build_demo_registry(workspace_service=None) -> ToolRegistry:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    notes = WORKSPACE_ROOT / "notes.txt"
    if not notes.exists():
        notes.write_text(
            "notes.txt — mock 工作区示例文件\n"
            "由 backend/dev_mock_server.py 启动时创建。\n"
            "用于演示 read_file 工具链路与前端 ToolCard 渲染。\n",
            encoding="utf-8",
        )
    registry = ToolRegistry()
    if workspace_service is not None:
        register_default_tools(
            registry, workspace_service.get_dirs, workspace_service=workspace_service
        )
    else:
        register_default_tools(registry, WORKSPACE_ROOT)
    from app.main import get_mcp_manager, get_tool_catalog, get_tools_broadcaster
    from app.tools.meta_tools import register_meta_tools

    catalog = get_tool_catalog()
    if catalog is not None:
        register_meta_tools(
            registry,
            catalog,
            mcp_provider=get_mcp_manager,
            on_change=get_tools_broadcaster(),
        )

    async def deploy_handler(tool_input: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {
            "status": "deployed",
            "env": tool_input.get("env"),
            "version": tool_input.get("version"),
        }

    registry.register(
        ToolSpec(
            name="deploy_tool",
            description="Deploy a release to an environment. DANGEROUS: requires approval.",
            input_schema={
                "type": "object",
                "properties": {
                    "env": {"type": "string"},
                    "version": {"type": "string"},
                },
                "required": ["env", "version"],
            },
            handler=deploy_handler,
            timeout_s=10,
            dangerous=True,
        )
    )
    return registry


def fake_build_session_manager(bus):
    from app.main import get_workspace_service

    settings = get_settings()
    registry = build_demo_registry(get_workspace_service())

    def dispatcher_factory(approve_callback):
        return ToolDispatcher(
            registry,
            approve_callback=approve_callback,
            breaker_limit=settings.tool_breaker_limit,
        )

    return SessionManager(
        session_factory=get_sessionmaker(),
        bus=bus,
        llm=FakeLLM(),
        registry=registry,
        dispatcher_factory=dispatcher_factory,
        system_prompt="You are a demo agent running against a scripted mock LLM.",
        model=settings.model,
        max_iterations=settings.max_iterations,
        max_tokens=settings.max_tokens,
        approval_timeout_s=settings.approval_timeout_s,
        idle_timeout_s=settings.session_idle_timeout_s,
    )


def create_mock_app():
    import app.main as main_mod
    from fastapi.responses import JSONResponse

    main_mod.build_session_manager = fake_build_session_manager

    real_app = main_mod.create_app()

    @real_app.post("/api/mock/drop_sse/{session_id}")
    async def drop_sse(session_id: str, request: Request) -> JSONResponse:
        manager = getattr(real_app.state, "sessions", None)
        if manager is None:
            return JSONResponse({"ok": False, "error": "manager not ready"}, status_code=503)
        state = manager.bus._states.get(session_id)
        if state is None:
            return JSONResponse({"ok": True, "session_id": session_id, "dropped": 0})
        from app.core.event_bus import _CLOSE, _push

        queues = list(state.subscribers)
        for queue in queues:
            _push(queue, _CLOSE)
        return JSONResponse(
            {"ok": True, "session_id": session_id, "dropped": len(queues)}
        )

    return real_app


app = create_mock_app()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("MOCK_PORT", "8000"))
    print("=" * 60)
    print("dev mock server: http://127.0.0.1:" + str(port))
    print("db:", os.environ["DATABASE_URL"])
    print("workspace:", WORKSPACE_ROOT.resolve())
    print("scenarios: chat(默认) file(读文件) approval(审批) slow(慢速) error(报错) workspace(多目录) tools(动态工具)")
    print("  env MOCK_SCENARIO 可固定场景；MOCK_DELTA_DELAY 调 delta 延迟")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
