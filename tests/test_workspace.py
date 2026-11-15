
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.dispatcher import ToolDispatcher
from app.core.session_manager import SessionManager
from app.core.workspace_registry import (
    DuplicateWorkspacePath,
    InvalidWorkspacePath,
    LastDirectoryProtected,
    WorkspaceDirNotFound,
    WorkspaceRegistry,
)
from app.db.database import build_engine, dispose_engine, get_sessionmaker
from app.db.models import Base
from app.db.repository import Repository
from app.schemas.events import SSEEventType
from app.tools.errors import ToolPermanentError
from app.tools.registry import ToolRegistry

from test_api import (
    ScriptedLLM,
    create_session,
    end_turn,
    manager_of,
    read_frames,
    tool_use_turn,
    wait_run_done,
)

SSE_PAYLOAD = "data"


@pytest.fixture
async def ws_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    yield factory
    await engine.dispose()


def make_service(
    factory: async_sessionmaker[AsyncSession],
    *,
    on_change=None,
) -> WorkspaceRegistry:
    return WorkspaceRegistry(factory, on_change=on_change)


@pytest.fixture
async def ws_api(tmp_path, monkeypatch):
    import app.main as main_mod

    db_path = (tmp_path / "ws_api_test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()

    scripted = ScriptedLLM()

    def fake_build_session_manager(bus):
        settings = get_settings()
        registry = ToolRegistry()

        def dispatcher_factory(approve_callback):
            return ToolDispatcher(
                registry, approve_callback=approve_callback, retry_backoff_base_s=0
            )

        return SessionManager(
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


@pytest.fixture
async def agent_api(tmp_path, monkeypatch):
    import app.main as main_mod

    db_path = (tmp_path / "ws_agent_test.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()

    scripted = ScriptedLLM()

    def fake_build_session_manager(bus):
        from app.main import get_workspace_service as current_service

        service = current_service()
        assert service is not None
        settings = get_settings()
        registry = ToolRegistry()
        from app.tools.builtin import register_default_tools

        register_default_tools(registry, service.get_dirs, workspace_service=service)

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


async def test_workspace_dir_crud_roundtrip(ws_factory) -> None:
    async with ws_factory() as db:
        repo = Repository(db)
        assert await repo.list_workspace_dirs() == []

        row, created = await repo.add_workspace_dir("D:\\data\\ws", source="user")
        assert created is True
        assert row.id and len(row.id) == 36
        assert row.source == "user"
        assert row.created_at is not None

        rows = await repo.list_workspace_dirs()
        assert [r.id for r in rows] == [row.id]

        fetched = await repo.get_workspace_dir_by_path("D:\\data\\ws")
        assert fetched is not None and fetched.id == row.id

        assert await repo.remove_workspace_dir(row.id) is True
        assert await repo.list_workspace_dirs() == []
        assert await repo.remove_workspace_dir(row.id) is False
        assert await repo.get_workspace_dir_by_path("D:\\data\\ws") is None


@pytest.mark.skipif(os.name != "nt", reason="大小写不敏感去重是 Windows normcase 语义")
async def test_workspace_dir_normcase_dedup_windows(ws_factory) -> None:
    async with ws_factory() as db:
        repo = Repository(db)
        row, created = await repo.add_workspace_dir("D:\\Data\\WS", source="user")
        assert created is True

        same, created2 = await repo.add_workspace_dir("d:\\data\\ws", source="agent")
        assert created2 is False
        assert same.id == row.id
        assert len(await repo.list_workspace_dirs()) == 1

        found = await repo.get_workspace_dir_by_path("D:\\data\\WS")
        assert found is not None and found.id == row.id


async def test_workspace_dir_source_values_roundtrip(ws_factory) -> None:
    async with ws_factory() as db:
        repo = Repository(db)
        _, _ = await repo.add_workspace_dir("C:\\a", source="user")
        b, _ = await repo.add_workspace_dir("C:\\b", source="agent")
        rows = await repo.list_workspace_dirs()
        assert [r.source for r in rows] == ["user", "agent"]
        assert rows[1].id == b.id


@pytest.fixture
def two_dirs(tmp_path: Path) -> tuple[Path, Path]:
    root1 = tmp_path / "ws_one"
    root1.mkdir()
    (root1 / "a.txt").write_text("alpha", encoding="utf-8")
    root2 = tmp_path / "ws_two"
    root2.mkdir()
    (root2 / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return root1, root2


async def test_registry_add_validates_absolute_existing_dir(
    ws_factory, tmp_path: Path
) -> None:
    service = make_service(ws_factory)
    await service.load()

    with pytest.raises(InvalidWorkspacePath) as ei:
        await service.add("relative/dir")
    assert ei.value.code == "invalid_path"

    with pytest.raises(InvalidWorkspacePath):
        await service.add(str(tmp_path / "missing"))

    plain_file = tmp_path / "file.txt"
    plain_file.write_text("x", encoding="utf-8")
    with pytest.raises(InvalidWorkspacePath):
        await service.add(str(plain_file))

    with pytest.raises(InvalidWorkspacePath):
        await service.add("   ")


async def test_registry_add_rejects_duplicate_normcase(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)

    record = await service.add(str(root2), source="agent")
    assert record.source == "agent"

    with pytest.raises(DuplicateWorkspacePath) as ei:
        await service.add(str(root2))
    assert ei.value.code == "duplicate_path"

    if os.name == "nt":
        with pytest.raises(DuplicateWorkspacePath):
            await service.add(str(root2).swapcase())


async def test_registry_last_directory_guard(ws_factory, tmp_path: Path) -> None:
    only = tmp_path / "only_one"
    only.mkdir()
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(only)
    assert len(service.get_dirs()) == 1

    with pytest.raises(LastDirectoryProtected) as ei:
        await service.remove(service.snapshot()[0].id)
    assert ei.value.code == "last_directory"
    assert len(service.get_dirs()) == 1


async def test_registry_remove_syncs_memory_and_db(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")
    assert len(service.get_dirs()) == 2

    agent_row = service.snapshot()[1]
    removed = await service.remove(agent_row.id)
    assert removed.id == agent_row.id
    assert [p.resolve() for p in service.get_dirs()] == [Path(root1).resolve()]

    async with ws_factory() as db:
        rows = await Repository(db).list_workspace_dirs()
    assert [r.path for r in rows] == [str(Path(root1).resolve())]

    with pytest.raises(WorkspaceDirNotFound):
        await service.remove(agent_row.id)


async def test_registry_remove_by_ref_id_or_path(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")

    removed = await service.remove_by_ref(str(root2))
    assert removed.path == str(Path(root2).resolve())

    await service.add(str(root2), source="user")
    removed = await service.remove_by_ref(service.snapshot()[1].id)
    assert removed.path == str(Path(root2).resolve())

    with pytest.raises(WorkspaceDirNotFound):
        await service.remove_by_ref("C:\\nowhere")


async def test_registry_on_change_callback_fires(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    events: list[tuple[str, str, str, list[str]]] = []

    async def on_change(action, record, dirs):
        events.append((action, record.path, record.source, [d.path for d in dirs]))

    service = make_service(ws_factory, on_change=on_change)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")
    removed = await service.remove(service.snapshot()[0].id)

    assert [e[0] for e in events] == ["add", "add", "remove"]
    assert events[1][1] == str(Path(root2).resolve())
    assert events[1][2] == "agent"
    assert events[2][0] == "remove" and events[2][1] == removed.path
    assert events[-1][3] == [d.path for d in service.snapshot()]


async def test_registry_seed_if_empty(ws_factory, tmp_path: Path) -> None:
    service = make_service(ws_factory)
    await service.load()
    assert service.is_empty()

    seeded = await service.seed_if_empty("./nonexistent/seed_dir")
    assert seeded is True
    assert service.get_dirs() == [Path("./nonexistent/seed_dir").resolve()]
    assert service.snapshot()[0].source == "user"

    assert await service.seed_if_empty("./other") is False
    assert len(service.get_dirs()) == 1


async def test_dynamic_binding_add_then_read_without_restart(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)

    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    specs = register_default_tools(registry, service.get_dirs, workspace_service=service)
    assert [s.name for s in specs] == [
        "add_workspace_dir",
        "calculate",
        "delete_file",
        "list_workspace_dirs",
        "read_file",
        "remove_workspace_dir",
        "run_command",
        "search_workspace",
        "write_file",
    ]
    assert all(s.dangerous is False for s in specs if s.access == "read")

    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("read_file").handler({"path": str(root2 / "b.txt")})
    assert ei.value.code == "path_outside_workspace"

    added = await registry.get("add_workspace_dir").handler({"path": str(root2)})
    assert added["dir"]["source"] == "agent"

    got = await registry.get("read_file").handler({"path": str(root2 / "b.txt")})
    assert got["content"] == "beta"
    got_rel = await registry.get("read_file").handler({"path": "a.txt"})
    assert got_rel["content"] == "alpha"

    found = await registry.get("search_workspace").handler({"pattern": "beta"})
    assert found["total"] == 1 and found["matches"][0]["file"] == "b.txt"

    listed = await registry.get("list_workspace_dirs").handler({})
    assert listed["count"] == 2
    assert {d["source"] for d in listed["dirs"]} == {"user", "agent"}


async def test_dynamic_binding_remove_then_outside_workspace(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")

    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    register_default_tools(registry, service.get_dirs, workspace_service=service)
    assert (await registry.get("read_file").handler({"path": str(root2 / "b.txt")}))["content"] == "beta"

    await registry.get("remove_workspace_dir").handler({"path_or_id": str(root2)})
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("read_file").handler({"path": str(root2 / "b.txt")})
    assert ei.value.code == "path_outside_workspace"

    found = await registry.get("search_workspace").handler({"pattern": "beta"})
    assert found["total"] == 0

    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("remove_workspace_dir").handler(
            {"path_or_id": service.snapshot()[0].id}
        )
    assert ei.value.code == "last_directory"


async def test_traversal_protection_spans_all_roots(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")

    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    register_default_tools(registry, service.get_dirs, workspace_service=service)

    got = await registry.get("read_file").handler({"path": str(root2 / "b.txt")})
    assert got["content"] == "beta"

    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("read_file").handler({"path": "../secret.txt"})
    assert ei.value.code == "path_outside_workspace"
    assert "workspace roots" in (ei.value.hint or "")

    with pytest.raises(ToolPermanentError):
        await registry.get("read_file").handler({"path": str(root1 / ".." / "secret.txt")})


async def test_empty_workspace_rejects_reads(ws_factory) -> None:
    service = make_service(ws_factory)
    await service.load()
    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    register_default_tools(registry, service.get_dirs, workspace_service=service)
    with pytest.raises(ToolPermanentError) as ei:
        await registry.get("read_file").handler({"path": "anything.txt"})
    assert ei.value.code == "path_outside_workspace"


@pytest.fixture
def manage_disabled(monkeypatch):
    monkeypatch.setenv("AGENT_MANAGE_DIRS", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_manage_tools_disabled_return_is_error(
    ws_factory, two_dirs: tuple[Path, Path], manage_disabled
) -> None:
    root1, root2 = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    await service.add(str(root2), source="agent")

    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    register_default_tools(registry, service.get_dirs, workspace_service=service)

    cases = [
        ("list_workspace_dirs", {}),
        ("add_workspace_dir", {"path": str(root2)}),
        ("remove_workspace_dir", {"path_or_id": str(root2)}),
    ]
    for name, payload in cases:
        with pytest.raises(ToolPermanentError) as ei:
            await registry.get(name).handler(payload)
        assert ei.value.code == "workspace_management_disabled"
        assert "workspace management disabled by config" in str(ei.value)

    got = await registry.get("read_file").handler({"path": str(root1 / "a.txt")})
    assert got["content"] == "alpha"


async def test_manage_tools_enabled_by_default(
    ws_factory, two_dirs: tuple[Path, Path]
) -> None:
    root1, _ = two_dirs
    service = make_service(ws_factory)
    await service.load()
    await service.seed_if_empty(root1)
    registry = ToolRegistry()
    from app.tools.builtin import register_default_tools

    register_default_tools(registry, service.get_dirs, workspace_service=service)
    result = await registry.get("list_workspace_dirs").handler({})
    assert result["count"] == 1


async def test_agent_autonomous_add_then_read_full_chain(agent_api, tmp_path: Path):
    client, application, scripted = agent_api
    import app.main as main_mod

    service = main_mod.get_workspace_service()
    assert service is not None

    agent_dir = tmp_path / "agent_ws"
    agent_dir.mkdir()
    (agent_dir / "target.txt").write_text("agent data", encoding="utf-8")

    session = await create_session(client)
    scripted.turns = [
        tool_use_turn("tu_add", "add_workspace_dir", {"path": str(agent_dir)}),
        tool_use_turn("tu_read", "read_file", {"path": str(agent_dir / "target.txt")}),
        end_turn("added and read"),
    ]
    resp = await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "把新目录加入工作区并读取"}
    )
    assert resp.status_code == 202
    await wait_run_done(application, session["id"])

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_results = [
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 2
    assert all(b["is_error"] is False for b in tool_results)
    assert "agent data" in tool_results[1]["content"]

    paths = {d.path: d.source for d in service.snapshot()}
    assert paths.get(str(agent_dir.resolve())) == "agent"

    manager = manager_of(application)
    buffer = manager.bus._states[session["id"]].buffer
    ws_events = [e for e in buffer if e.type == SSEEventType.WORKSPACE_CHANGED]
    assert len(ws_events) == 1
    assert ws_events[0].data["action"] == "add"
    assert ws_events[0].data["source"] == "agent"
    assert any(
        d["path"] == str(agent_dir.resolve()) for d in ws_events[0].data["dirs"]
    )


async def test_agent_remove_guard_via_tool_chain(agent_api, tmp_path: Path):
    client, application, scripted = agent_api
    import app.main as main_mod

    service = main_mod.get_workspace_service()
    session = await create_session(client)
    seed_id = service.snapshot()[0].id
    scripted.turns = [
        tool_use_turn("tu_rm", "remove_workspace_dir", {"path_or_id": seed_id}),
        end_turn("cannot remove the last one"),
    ]
    await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "删掉工作区"}
    )
    await wait_run_done(application, session["id"])

    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    tool_result = next(
        b for m in messages for b in m["content"] if b.get("type") == "tool_result"
    )
    assert tool_result["is_error"] is True
    assert "last_directory" in tool_result["content"]
    assert len(service.snapshot()) == 1


async def test_workspaces_api_list_seeded(ws_api):
    client, _, _ = ws_api
    resp = await client.get("/api/workspaces")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["dirs"]) == 1
    seed = body["dirs"][0]
    assert seed["source"] == "user"
    assert seed["path"] == str(Path(get_settings().workspace_root).resolve())
    assert seed["id"] and seed["created_at"]


async def test_workspaces_api_add_and_errors(ws_api, tmp_path):
    client, _, _ = ws_api

    new_dir = tmp_path / "api_ws"
    new_dir.mkdir()
    resp = await client.post("/api/workspaces", json={"path": str(new_dir)})
    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True
    assert body["dir"]["path"] == str(new_dir.resolve())
    assert body["dir"]["source"] == "user"
    assert set(body["dir"]) == {"id", "path", "source", "created_at"}

    listed = (await client.get("/api/workspaces")).json()["dirs"]
    assert {d["path"] for d in listed} >= {str(new_dir.resolve())}

    resp = await client.post("/api/workspaces", json={"path": "relative/dir"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "invalid_path"
    assert body["error"]["message"]

    resp = await client.post("/api/workspaces", json={"path": str(tmp_path / "ghost")})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_path"

    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    resp = await client.post("/api/workspaces", json={"path": str(plain)})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_path"

    resp = await client.post("/api/workspaces", json={"path": str(new_dir)})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "duplicate_path"

    if os.name == "nt":
        resp = await client.post("/api/workspaces", json={"path": str(new_dir).swapcase()})
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "duplicate_path"


async def test_workspaces_api_delete_and_guards(ws_api, tmp_path):
    client, _, _ = ws_api

    resp = await client.delete("/api/workspaces/no-such-id")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"

    extra = tmp_path / "extra_ws"
    extra.mkdir()
    created = (await client.post("/api/workspaces", json={"path": str(extra)})).json()["dir"]

    resp = await client.delete(f"/api/workspaces/{created['id']}")
    assert resp.status_code == 204
    listed = (await client.get("/api/workspaces")).json()["dirs"]
    assert created["id"] not in {d["id"] for d in listed}

    seed = listed[0]
    resp = await client.delete(f"/api/workspaces/{seed['id']}")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "last_directory"
    assert (await client.get("/api/workspaces")).json()["dirs"]

    resp = await client.delete(f"/api/workspaces/{created['id']}")
    assert resp.status_code == 404


async def test_workspace_changed_broadcast_over_sse(ws_api, tmp_path):
    client, application, scripted = ws_api
    manager = manager_of(application)

    session = await create_session(client)
    scripted.turns = [end_turn("warm")]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "warm"})
    await wait_run_done(application, session["id"])

    last_seq = manager.bus.last_seq(session["id"])
    frames_task = asyncio.create_task(
        read_frames(
            application,
            f"/api/sessions/{session['id']}/events?after_seq={last_seq}",
            count=1,
        )
    )
    await asyncio.sleep(0.05)

    new_dir = tmp_path / "broadcast_ws"
    new_dir.mkdir()
    resp = await client.post("/api/workspaces", json={"path": str(new_dir)})
    assert resp.status_code == 201

    _, _, frames = await frames_task
    kind, envelope = frames[0]
    assert kind == "workspace_changed"
    assert envelope["type"] == "workspace_changed"
    payload = envelope[SSE_PAYLOAD]
    assert payload["action"] == "add"
    assert payload["path"] == str(new_dir.resolve())
    assert payload["source"] == "user"
    assert {d["path"] for d in payload["dirs"]} >= {str(new_dir.resolve())}

    dirs_now = (await client.get("/api/workspaces")).json()["dirs"]
    target = next(d for d in dirs_now if d["path"] == str(new_dir.resolve()))
    frames_task = asyncio.create_task(
        read_frames(
            application,
            f"/api/sessions/{session['id']}/events?after_seq={manager.bus.last_seq(session['id'])}",
            count=1,
        )
    )
    await asyncio.sleep(0.05)
    assert (await client.delete(f"/api/workspaces/{target['id']}")).status_code == 204
    _, _, frames = await frames_task
    assert frames[0][0] == "workspace_changed"
    assert frames[0][1][SSE_PAYLOAD]["action"] == "remove"


async def test_workspace_changed_reaches_cold_session(ws_api, tmp_path):
    client, application, _ = ws_api
    manager = manager_of(application)

    session = await create_session(client)
    assert manager.get_running(session["id"]) is None

    frames_task = asyncio.create_task(
        read_frames(
            application,
            f"/api/sessions/{session['id']}/events?after_seq=0",
            count=1,
        )
    )
    await asyncio.sleep(0.05)

    new_dir = tmp_path / "cold_ws"
    new_dir.mkdir()
    resp = await client.post("/api/workspaces", json={"path": str(new_dir)})
    assert resp.status_code == 201

    _, _, frames = await frames_task
    assert frames[0][0] == "workspace_changed"
    payload = frames[0][1][SSE_PAYLOAD]
    assert payload["action"] == "add"
    assert payload["path"] == str(new_dir.resolve())


async def test_workspace_tool_error_broadcasts_nothing(ws_api, tmp_path):
    client, application, scripted = ws_api
    manager = manager_of(application)
    session = await create_session(client)
    scripted.turns = [end_turn("warm")]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "warm"})
    await wait_run_done(application, session["id"])

    seed = (await client.get("/api/workspaces")).json()["dirs"][0]
    resp = await client.post("/api/workspaces", json={"path": seed["path"]})
    assert resp.status_code == 409

    buffer = manager.bus._states[session["id"]].buffer
    assert not any(e.type == SSEEventType.WORKSPACE_CHANGED for e in buffer)


async def test_lifespan_seeds_and_reuses_existing_rows(ws_api):
    client, application, _ = ws_api
    before = (await client.get("/api/workspaces")).json()["dirs"]
    assert len(before) == 1

    service = application.state.workspaces
    await service.load()
    assert await service.seed_if_empty(get_settings().workspace_root) is False
    after = (await client.get("/api/workspaces")).json()["dirs"]
    assert [d["id"] for d in after] == [d["id"] for d in before]
