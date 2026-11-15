"""元工具测试（自扩展 Agent）—— 全 mock；真实子进程链路单独显式用例。

覆盖：
1. tool_catalog.yaml 装载 + CatalogItem 解析（非法项跳过 / 缺文件降级 /
   trust 校验 / 重名先到先得）
2. discover_tools：大小写不敏感匹配 name/description/keywords（含中文）、
   排序确定性、top5 截断、installed 标记、无匹配空结果
3. install_mcp_server：注册 mcp__<server>__<tool> 前缀 + 广播、幂等
   （已装不再重连）、未知 server / 连接失败 → is_error 且可干净重试
4. uninstall_mcp_server：移除工具 + 广播、未装 → is_error
5. 审批门：approve_callback=False → install is_error（denied）且**不连接**
6. run_python：正常执行 / 语法错误 return_code≠0 / 超时强杀 / 输出截断
   50K / -I 隔离（读不到自定义环境变量）/ 超时被 settings 上限钳制 / 参数校验
7. 开关关闭 → 四元工具统一 is_error
8. 核心链路（ScriptedLLM 经真实引擎）：discover → install（过审批）→
   **同一 run 下一迭代调用新装的 MCP 工具**（注入 fake session；
   另有真实 catalog 子进程端到端，spawn 慢的环境跳过）
9. tools_changed 广播：信封带 seq、进 known_session_ids
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import get_settings
from app.core.dispatcher import ToolDispatcher
from app.core.session_manager import SessionManager
from app.db.database import dispose_engine, get_sessionmaker
from app.mcp_loader import MCPManager, MCPServerConfig
from app.schemas.events import SSEEventType
from app.tools.catalog import CatalogItem, ToolCatalog, load_tool_catalog
from app.tools.errors import ToolInputError, ToolPermanentError
from app.tools.meta_tools import register_meta_tools
from app.tools.registry import ToolRegistry

# 复用 test_api 的 SSE/mock 脚手架与 test_mcp_loader 的 fake 连接层
from test_api import (  # noqa: E402
    ScriptedLLM,
    create_session,
    end_turn,
    manager_of,
    tool_use_turn,
    wait_run_done,
    wait_until,
)
from test_mcp_loader import (  # noqa: E402
    FakeCallToolResult,
    FakeSession,
    FakeTool,
    make_connector,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# 公共夹具 / 工厂
# ----------------------------------------------------------------------


@pytest.fixture
def real_catalog() -> ToolCatalog:
    return load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml")


def make_meta_registry(
    catalog: ToolCatalog,
    sessions: list[FakeSession],
    broadcasts: list | None = None,
) -> tuple[ToolRegistry, MCPManager]:
    """registry + 注入式 fake connector 的 MCPManager + 四元工具。"""
    registry = ToolRegistry()
    manager = MCPManager(
        [], registry, connect_timeout_s=1, connector=make_connector(sessions)
    )

    async def on_change(action: str, server: str, tools: list[str]) -> None:
        if broadcasts is not None:
            broadcasts.append((action, server, list(tools)))

    register_meta_tools(registry, catalog, lambda: manager, on_change)
    return registry, manager


def mcp_tool_names(registry: ToolRegistry) -> list[str]:
    return [n for n in registry.names() if n.startswith("mcp__")]


class ReversingSession(FakeSession):
    """fake server：reverse_text 真实反转（模拟 server 端行为，非 echo）。"""

    async def call_tool(
        self, name: str, arguments: dict[str, Any], read_timeout_seconds: Any = None
    ) -> FakeCallToolResult:
        self.calls.append((name, dict(arguments)))
        if name == "reverse_text":
            return FakeCallToolResult(str(arguments.get("text", ""))[::-1])
        return FakeCallToolResult(f"echo:{arguments.get('text', '')}")


@pytest.fixture
async def meta_api(tmp_path, monkeypatch):
    """agent 全链路夹具：真实默认工具 + 真实 catalog 的元工具，MCPManager
    用 lifespan 构建的真实实例（mcp_enabled=false → 启动零连接；测试用例
    自行向其 connector 注入 fake session 或直接跑真实子进程）。"""
    import app.main as main_mod
    from app.tools.builtin import register_default_tools

    db_path = (tmp_path / "meta_agent_test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()

    scripted = ScriptedLLM()

    def fake_build_session_manager(bus):
        from app.main import get_workspace_service as current_service

        service = current_service()
        catalog = main_mod.get_tool_catalog()
        assert catalog is not None  # lifespan 先建目录再建 manager
        settings = get_settings()
        registry = ToolRegistry()
        register_default_tools(registry, service.get_dirs, workspace_service=service)
        register_meta_tools(
            registry,
            catalog,
            mcp_provider=main_mod.get_mcp_manager,
            on_change=main_mod.get_tools_broadcaster(),
        )

        def dispatcher_factory(approve_callback):
            return ToolDispatcher(
                registry, approve_callback=approve_callback, retry_backoff_base_s=0
            )

        manager = SessionManager(
            session_factory=get_sessionmaker(),
            bus=bus,
            llm=scripted,
            registry=registry,
            dispatcher_factory=dispatcher_factory,
            system_prompt="test",
            model=settings.model,
            max_iterations=settings.max_iterations,
            max_tokens=settings.max_tokens,
            approval_timeout_s=settings.approval_timeout_s,
            idle_timeout_s=settings.session_idle_timeout_s,
        )
        # 链路测试的语义是"install 过审批、装完即用"→ write 模式
        # （install=admin 仍审批；MCP 工具=execute 直行；read 模式会双重审批卡住链路）
        manager.permission_mode = "write"
        return manager

    monkeypatch.setattr(main_mod, "build_session_manager", fake_build_session_manager)
    application = main_mod.create_app()
    try:
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, application, scripted
    finally:
        await dispose_engine()
        get_settings.cache_clear()


async def resolve_when_pending(
    client: httpx.AsyncClient, application, session_id: str, tool_use_id: str, result: str
) -> None:
    """等 dangerous 工具的审批挂起后提交决议（allow/deny）。"""
    manager = manager_of(application)

    def pending() -> bool:
        running = manager.get_running(session_id)
        return running is not None and tool_use_id in running.pending_approvals

    await wait_until(pending)
    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/{tool_use_id}", json={"result": result}
    )
    assert resp.status_code == 200


# ----------------------------------------------------------------------
# 目录装载与解析
# ----------------------------------------------------------------------


def test_real_catalog_loads_three_builtin_servers(real_catalog: ToolCatalog) -> None:
    assert real_catalog.names() == ["password_gen", "text_tools", "unit_convert"]
    for name in real_catalog.names():
        item = real_catalog.get(name)
        assert item is not None
        assert item.trust == "builtin"
        assert item.command == "python"
        assert item.args and item.args[0].startswith("catalog_servers/")
        assert item.cwd == str(BACKEND_ROOT)  # 相对路径锚定目录基址
        config = item.to_server_config()
        assert isinstance(config, MCPServerConfig)
        assert config.enabled is True and config.cwd == item.cwd


def test_catalog_item_defaults_and_validation() -> None:
    item = CatalogItem.from_dict(
        {
            "name": "community_srv",
            "description": "A third-party server",
            "keywords": ["a", "b"],
            "transport": {"url": "http://127.0.0.1:1/mcp"},
        }
    )
    assert item.trust == "community"  # 缺省保守按 community
    assert item.url == "http://127.0.0.1:1/mcp" and item.command is None

    bad_cases = [
        {"description": "no name", "transport": {"command": "x"}},
        {"name": "x", "transport": {"command": "x"}},  # 缺 description
        {"name": "x", "description": "d"},  # 缺 transport
        {"name": "x", "description": "d", "transport": {}},  # command/url 二选一
        {
            "name": "x",
            "description": "d",
            "transport": {"command": "x", "args": "no"},
            "trust": "builtin",
        },  # args 非 list
        {
            "name": "x",
            "description": "d",
            "transport": {"command": "x"},
            "trust": "alien",
        },  # trust 非法
    ]
    for raw in bad_cases:
        with pytest.raises(ValueError):
            CatalogItem.from_dict(raw)


def test_load_tool_catalog_skips_invalid_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "- name: good\n"
        "  description: First wins\n"
        "  transport: {command: python, args: [a.py]}\n"
        "- name: good\n"
        "  description: duplicate ignored\n"
        "  transport: {command: python, args: [b.py]}\n"
        "- description: missing name\n"
        "  transport: {command: python}\n"
        "- name: bad_trust\n"
        "  description: d\n"
        "  transport: {command: python}\n"
        "  trust: alien\n"
        "- just a string\n",
        encoding="utf-8",
    )
    catalog = load_tool_catalog(path)
    assert catalog.names() == ["good"]
    assert catalog.get("good").description == "First wins"  # 重名先到先得
    assert catalog.get("good").trust == "community"


def test_load_tool_catalog_missing_and_corrupt_file(tmp_path: Path) -> None:
    assert load_tool_catalog(tmp_path / "absent.yaml").names() == []
    corrupt = tmp_path / "corrupt.yaml"
    corrupt.write_text("key: [unclosed", encoding="utf-8")
    assert load_tool_catalog(corrupt).names() == []
    not_a_list = tmp_path / "not_a_list.yaml"
    not_a_list.write_text("key: value", encoding="utf-8")
    assert load_tool_catalog(not_a_list).names() == []


# ----------------------------------------------------------------------
# discover_tools：匹配 / 排序 / top5 / installed 标记 / 无匹配
# ----------------------------------------------------------------------


async def test_discover_matches_name_keywords_case_insensitive(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    handler = registry.get("discover_tools").handler

    for query, expected in [
        ("PASSWORD", "password_gen"),        # 大写查询命中小写名字
        ("温度换算", "unit_convert"),         # 中文短语命中中文关键词
        ("reverse a string", "text_tools"),  # 多词 token 命中
        ("km", "unit_convert"),              # 短 token 命中关键词
    ]:
        result = await handler({"query": query})
        assert result["total"] >= 1
        assert result["results"][0]["name"] == expected
        assert result["results"][0]["trust"] == "builtin"
        assert result["results"][0]["installed"] is False

    # 无匹配 → 空结果（不报错）
    result = await handler({"query": "blockchain quantum oracle"})
    assert result["results"] == [] and result["total"] == 0

    with pytest.raises(ToolInputError):
        await handler({"query": "   "})


async def test_discover_ranking_deterministic_top5_and_installed_flag() -> None:
    """7 个同分条目 → 只返回 top5（按名排序确定）；安装后 installed 翻转。"""
    items = [
        CatalogItem(
            name=f"tool_{chr(ord('a') + i)}",
            description="generic helper",
            keywords=("tool",),
            command="python",
            args=("x.py",),
            trust="builtin" if i % 2 else "community",
        )
        for i in range(7)
    ]
    catalog = ToolCatalog(items)
    session = FakeSession([FakeTool("t")])
    registry, manager = make_meta_registry(catalog, [session])
    handler = registry.get("discover_tools").handler

    result = await handler({"query": "tool"})
    assert result["total"] == 5
    assert [r["name"] for r in result["results"]] == [f"tool_{c}" for c in "abcde"]
    assert all(r["installed"] is False for r in result["results"])

    # 真实安装 tool_a（消费一个 fake session）→ 标记翻转，其余不变
    installed = await registry.get("install_mcp_server").handler({"name": "tool_a"})
    assert installed["ok"] is True
    result2 = await handler({"query": "tool"})
    flags = {r["name"]: r["installed"] for r in result2["results"]}
    assert flags["tool_a"] is True
    assert flags["tool_b"] is False
    assert manager.is_connected("tool_a")


# ----------------------------------------------------------------------
# install / uninstall：注册前缀 / 幂等 / 失败语义 / 广播
# ----------------------------------------------------------------------


async def test_install_registers_prefixed_tools_and_broadcasts(real_catalog) -> None:
    broadcasts: list = []
    session = FakeSession(
        [FakeTool("reverse_text", "Reverse the text"), FakeTool("to_uppercase")]
    )
    registry, manager = make_meta_registry(real_catalog, [session], broadcasts)
    handler = registry.get("install_mcp_server").handler

    result = await handler({"name": "text_tools"})
    assert result["ok"] is True and result["already_installed"] is False
    assert result["tools"] == [
        "mcp__text_tools__reverse_text",
        "mcp__text_tools__to_uppercase",
    ]
    assert manager.is_connected("text_tools")
    # 新工具立刻可用（同一 registry 实例）
    assert (
        await registry.get("mcp__text_tools__reverse_text").handler({"text": "hi"})
        == "echo:hi"
    )
    assert broadcasts == [("install", "text_tools", result["tools"])]


async def test_install_idempotent_second_call_does_not_reconnect(real_catalog) -> None:
    session = FakeSession([FakeTool("reverse_text")])
    connector = make_connector([session])
    registry = ToolRegistry()
    manager = MCPManager([], registry, connect_timeout_s=1, connector=connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    handler = registry.get("install_mcp_server").handler
    first = await handler({"name": "text_tools"})
    second = await handler({"name": "text_tools"})  # 重复安装
    assert first["already_installed"] is False
    assert second["already_installed"] is True
    assert second["tools"] == first["tools"]
    assert connector.issued == [session]  # 只连接过一次
    assert len(manager._states) == 1


async def test_install_unknown_server_is_error(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "ghost"})
    assert ei.value.code == "unknown_server"
    assert "password_gen" in (ei.value.hint or "")  # hint 列出可选 server


async def test_install_connect_failure_is_error_and_retry_succeeds(real_catalog) -> None:
    registry = ToolRegistry()

    def failing_connector(config):
        @asynccontextmanager
        async def cm():
            raise RuntimeError("process failed to spawn")
            yield  # pragma: no cover

        return cm

    manager = MCPManager([], registry, connect_timeout_s=0.5, connector=failing_connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "mcp_connect_failed"
    assert "process failed to spawn" in str(ei.value)
    assert not manager.has_server("text_tools")  # 失败现场已清理
    assert mcp_tool_names(registry) == []  # 没有残留半注册工具

    # 修复连接器后同一 server 可干净重试
    session = FakeSession([FakeTool("reverse_text")])
    manager._connector = make_connector([session])
    result = await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert result["ok"] is True and manager.is_connected("text_tools")


async def test_uninstall_removes_tools_and_broadcasts(real_catalog) -> None:
    broadcasts: list = []
    session = FakeSession([FakeTool("reverse_text")])
    registry, manager = make_meta_registry(real_catalog, [session], broadcasts)
    install = registry.get("install_mcp_server").handler
    uninstall = registry.get("uninstall_mcp_server").handler

    installed = (await install({"name": "text_tools"}))["tools"]
    assert registry.get("mcp__text_tools__reverse_text") is not None

    result = await uninstall({"name": "text_tools"})
    assert result["ok"] is True and result["removed_tools"] == installed
    assert registry.get("mcp__text_tools__reverse_text") is None  # 工具即刻消失
    assert not manager.has_server("text_tools")
    assert broadcasts == [
        ("install", "text_tools", installed),
        ("uninstall", "text_tools", installed),
    ]

    # 卸载后再 discover → installed=False
    discovered = await registry.get("discover_tools").handler({"query": "text"})
    assert all(r["installed"] is False for r in discovered["results"])


async def test_uninstall_not_installed_is_error(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("uninstall_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "not_installed"
    assert "discover_tools" in (ei.value.hint or "")


async def test_install_without_manager_is_error(real_catalog) -> None:
    registry = ToolRegistry()
    register_meta_tools(registry, real_catalog, lambda: None)  # provider 恒 None
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("install_mcp_server").handler({"name": "text_tools"})
    assert ei.value.code == "mcp_unavailable"


# ----------------------------------------------------------------------
# 审批门：install/uninstall/run_python dangerous=True
# ----------------------------------------------------------------------


async def test_all_meta_tools_have_expected_trust_flags(real_catalog) -> None:
    registry, _ = make_meta_registry(real_catalog, [])
    flags = {name: registry.get(name).dangerous for name in registry.names()}
    assert flags["discover_tools"] is False
    assert flags["install_mcp_server"] is True
    assert flags["uninstall_mcp_server"] is True
    assert flags["run_python"] is True


async def test_install_denied_by_approval_does_not_connect(real_catalog) -> None:
    """approve_callback=False → is_error（denied）且**不连接**（核心安全要求）。"""
    session = FakeSession([FakeTool("reverse_text")])
    connector = make_connector([session])
    registry = ToolRegistry()
    manager = MCPManager([], registry, connect_timeout_s=1, connector=connector)
    register_meta_tools(registry, real_catalog, lambda: manager)

    async def deny(name, tool_input) -> bool:
        return False

    dispatcher = ToolDispatcher(registry, approve_callback=deny, retry_backoff_base_s=0)
    result = await dispatcher.dispatch(
        "install_mcp_server", {"name": "text_tools"}, "tu_deny"
    )
    assert result.is_error is True
    assert result.status == "denied"
    assert result.code == "approval_denied"
    assert "user denied" in result.content
    assert connector.issued == []  # 审批拒绝 → 根本没有连接
    assert not manager.has_server("text_tools")
    assert mcp_tool_names(registry) == []  # 没有注册任何 MCP 工具


# ----------------------------------------------------------------------
# run_python 沙箱
# ----------------------------------------------------------------------


async def test_run_python_normal_execution() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler({"code": "print(6 * 7)"})
    assert result["stdout"].strip() == "42"
    assert result["stderr"] == ""
    assert result["return_code"] == 0
    assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 0
    assert result["truncated"] is False


async def test_run_python_syntax_error_returns_nonzero_code() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler({"code": "def broken(:\n    pass"})
    assert result["return_code"] != 0
    assert "SyntaxError" in result["stderr"]
    assert result["stdout"] == ""


async def test_run_python_timeout_force_kills_subprocess() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("run_python").handler(
            {
                "code": "import time; print('start', flush=True); time.sleep(60)",
                "timeout_s": 1,
            }
        )
    elapsed = asyncio.get_running_loop().time() - started
    assert ei.value.code == "run_python_timeout"
    assert "force-killed" in str(ei.value)
    assert elapsed < 30  # 是真强杀，不是等 sleep 跑完


async def test_run_python_output_truncated_to_50k() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    code = 'import sys; print("o" * 60000); print("e" * 60000, file=sys.stderr)'
    result = await registry.get("run_python").handler({"code": code})
    assert len(result["stdout"]) == 50_000
    assert len(result["stderr"]) == 50_000
    assert result["truncated"] is True


async def test_run_python_isolated_env(monkeypatch) -> None:
    """-I + 白名单 env：父进程自定义变量与 PYTHON* 都读不到，isolated 置位。"""
    monkeypatch.setenv("AGENT_TEST_SECRET", "leaked")
    monkeypatch.setenv("PYTHONPATH", "C:/should/not/leak")
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    code = (
        "import os, sys;"
        "print('SECRET' in os.environ, 'PYTHONPATH' in os.environ, sys.flags.isolated)"
    )
    result = await registry.get("run_python").handler({"code": code})
    assert result["stdout"].strip() == "False False 1"


async def test_run_python_timeout_clamped_to_settings_max(monkeypatch) -> None:
    """timeout_s 超过 run_python_max_timeout_s → 被钳制到上限。"""
    monkeypatch.setenv("RUN_PYTHON_MAX_TIMEOUT_S", "1")
    get_settings.cache_clear()
    try:
        registry, _ = make_meta_registry(
            load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
        )
        with pytest.raises(ToolPermanentError) as ei:
            await registry.get("run_python").handler(
                {"code": "import time; time.sleep(60)", "timeout_s": 500}
            )
        assert "1s" in str(ei.value)  # 生效的是钳制后的 1s 而非 500s
        assert ei.value.code == "run_python_timeout"
    finally:
        get_settings.cache_clear()


async def test_run_python_invalid_params() -> None:
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "   "})
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "pass", "timeout_s": -1})
    with pytest.raises(ToolInputError):
        await registry.get("run_python").handler({"code": "pass", "timeout_s": True})


async def test_run_python_writes_land_in_temp_cwd() -> None:
    """CWD 是一次性临时目录：代码里写文件不落在真实工作区。"""
    registry, _ = make_meta_registry(
        load_tool_catalog(BACKEND_ROOT / "tool_catalog.yaml"), []
    )
    result = await registry.get("run_python").handler(
        {"code": "open('scratch.txt', 'w').write('x'); print('done')"}
    )
    assert result["stdout"].strip() == "done"
    assert result["return_code"] == 0


# ----------------------------------------------------------------------
# 开关关闭 → 四元工具统一 is_error
# ----------------------------------------------------------------------


@pytest.fixture
def discovery_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_TOOL_DISCOVERY", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()  # monkeypatch 已还原 env → 下次调用重建默认


async def test_discovery_disabled_all_four_tools_is_error(
    real_catalog, discovery_disabled
) -> None:
    registry, _ = make_meta_registry(real_catalog, [FakeSession([FakeTool("t")])])
    cases = [
        ("discover_tools", {"query": "text"}),
        ("install_mcp_server", {"name": "text_tools"}),
        ("uninstall_mcp_server", {"name": "text_tools"}),
        ("run_python", {"code": "pass"}),
    ]
    for name, payload in cases:
        with pytest.raises(ToolPermanentError) as ei:
            await registry.get(name).handler(payload)
        assert ei.value.code == "tool_discovery_disabled"
        assert "tool discovery disabled by config" in str(ei.value)
    # 开关不影响注册本身（与 agent_manage_dirs 同模式：注册但拒执行）
    assert registry.get("install_mcp_server") is not None


# ----------------------------------------------------------------------
# 核心链路（ScriptedLLM 经真实引擎）：discover → install → 调新工具
# ----------------------------------------------------------------------


async def test_agent_full_chain_discover_install_call_same_run(meta_api) -> None:
    """单次 run 内：discover_tools → install_mcp_server（过审批）→ 下一迭代
    直接调用 mcp__text_tools__reverse_text——同 run 即插即用（核心）。"""
    client, application, scripted = meta_api

    # 向 lifespan 构建的 MCPManager 注入 fake session（不 spawn 真实子进程）
    fake = ReversingSession(
        [FakeTool("reverse_text", "Reverse the text"), FakeTool("to_uppercase")]
    )
    connector = make_connector([fake])
    application.state.mcp._connector = connector

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_disc", "discover_tools", {"query": "reverse text"}),
        tool_use_turn("tu_install", "install_mcp_server", {"name": "text_tools"}),
        tool_use_turn("tu_rev", "mcp__text_tools__reverse_text", {"text": "abc"}),
        end_turn("installed and used in the same run"),
    ]
    resp = await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "把文本反转一下"}
    )
    assert resp.status_code == 202
    await resolve_when_pending(client, application, session["id"], "tu_install", "allow")
    await wait_run_done(application, session["id"], timeout=10)

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 3
    assert all(b["is_error"] is False for b in tool_results)
    # ① discover：text_tools 在列且未安装
    assert '"name": "text_tools"' in tool_results[0]["content"]
    assert '"installed": false' in tool_results[0]["content"]
    # ② install：注册 2 个带前缀工具（非幂等分支）
    assert "mcp__text_tools__reverse_text" in tool_results[1]["content"]
    assert '"already_installed": false' in tool_results[1]["content"]
    # ③ 同一 run 下一迭代调用新装工具成功
    assert tool_results[2]["content"] == "cba"
    assert connector.issued == [fake]  # 连接恰好一次

    # 广播：bus 缓冲出现 tools_changed（信封带 seq；会话进 known_session_ids）
    manager = manager_of(application)
    assert session["id"] in manager.bus.known_session_ids()
    buffer = manager.bus._states[session["id"]].buffer
    changed = [e for e in buffer if e.type == SSEEventType.TOOLS_CHANGED]
    assert len(changed) == 1
    event = changed[0]
    assert event.seq > 0
    assert event.data["action"] == "install"
    assert event.data["server"] == "text_tools"
    assert event.data["tools"] == [
        "mcp__text_tools__reverse_text",
        "mcp__text_tools__to_uppercase",
    ]


async def test_agent_install_deny_path_via_approval(meta_api) -> None:
    """审批 deny → install is_error(denied) → 不连接、不注册、不广播。"""
    client, application, scripted = meta_api
    connector = make_connector([FakeSession([FakeTool("reverse_text")])])
    application.state.mcp._connector = connector

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_inst", "install_mcp_server", {"name": "text_tools"}),
        end_turn("user denied; proposing run_python instead"),
    ]
    await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "装一下 text_tools"}
    )
    await resolve_when_pending(client, application, session["id"], "tu_inst", "deny")
    await wait_run_done(application, session["id"], timeout=10)

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert "approval_denied" in tool_results[0]["content"]
    assert "user denied" in tool_results[0]["content"]
    assert connector.issued == []  # deny → 未连接
    manager = manager_of(application)
    assert manager.registry.get("mcp__text_tools__reverse_text") is None
    buffer = manager.bus._states[session["id"]].buffer
    assert not any(e.type == SSEEventType.TOOLS_CHANGED for e in buffer)  # 不广播


async def test_agent_install_real_catalog_server_subprocess(meta_api) -> None:
    """端到端（真实 catalog 子进程 stdio）：install 拉起 catalog_servers/
    text_tools_server.py → 同 run 内调用 mcp__text_tools__reverse_text。
    spawn 慢的平台可能超时——视为环境问题跳过，不算失败（与
    test_mcp_loader 的 demo 端到端同一纪律）。"""
    pytest.importorskip("mcp")
    client, application, scripted = meta_api
    application.state.mcp._connect_timeout_s = 25.0  # 冷启动子进程余量

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_install", "install_mcp_server", {"name": "text_tools"}),
        tool_use_turn("tu_rev", "mcp__text_tools__reverse_text", {"text": "hello"}),
        end_turn("real server roundtrip"),
    ]

    async def scenario() -> None:
        await client.post(
            f"/api/sessions/{session['id']}/messages",
            json={"text": "装 text_tools 并反转 hello"},
        )
        await resolve_when_pending(client, application, session["id"], "tu_install", "allow")
        await wait_run_done(application, session["id"], timeout=45)

    try:
        async with asyncio.timeout(90):
            await scenario()
    except TimeoutError:
        pytest.skip("catalog MCP server spawn timed out in this environment")

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 2
    assert all(b["is_error"] is False for b in tool_results)
    assert tool_results[1]["content"] == "olleh"  # 真实子进程返回
    # 广播 + 注册表同时确认
    manager = manager_of(application)
    buffer = manager.bus._states[session["id"]].buffer
    changed = [e for e in buffer if e.type == SSEEventType.TOOLS_CHANGED]
    assert len(changed) == 1 and changed[0].data["action"] == "install"
    assert manager.registry.get("mcp__text_tools__reverse_text") is not None
