
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

from app.core.config import get_settings
from app.core.llm_client import LLMTurnResult, StreamDelta, UsageData
from app.db.database import dispose_engine, get_sessionmaker


class ScriptedLLM:

    def __init__(self) -> None:
        self.turns: list[Any] = []
        self.deltas: list[list[StreamDelta]] = []
        self.calls: list[dict[str, Any]] = []

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.calls.append({k: deepcopy(v) for k, v in kwargs.items() if k != "on_delta"})
        on_delta = kwargs.get("on_delta")
        if self.deltas:
            for delta in self.deltas.pop(0):
                if on_delta is not None:
                    await on_delta(delta)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class BlockingLLM:

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run_turn(self, **kwargs: Any) -> LLMTurnResult:
        self.started.set()
        await self.release.wait()
        return LLMTurnResult(
            content=[{"type": "text", "text": "unblocked"}],
            stop_reason="end_turn",
            usage=UsageData(10, 5),
        )


def end_turn(text: str = "done") -> LLMTurnResult:
    return LLMTurnResult(
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=UsageData(150, 10, 140),
    )


def tool_use_turn(tid: str, tool_name: str, tool_input: dict) -> LLMTurnResult:
    return LLMTurnResult(
        content=[
            {"type": "text", "text": "working"},
            {"type": "tool_use", "id": tid, "name": tool_name, "input": tool_input},
        ],
        stop_reason="tool_use",
        usage=UsageData(100, 20, 90),
    )


@pytest.fixture
async def api(tmp_path, monkeypatch):
    import app.main as main_mod
    from app.core.dispatcher import ToolDispatcher
    from app.core.session_manager import SessionManager
    from app.tools.registry import ToolRegistry

    db_path = (tmp_path / "api_test.db").as_posix()
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


def manager_of(application) -> Any:
    return application.state.sessions


async def wait_until(pred, timeout: float = 3.0, interval: float = 0.02) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


async def wait_run_done(application, session_id: str, timeout: float = 5.0) -> None:
    manager = manager_of(application)
    await wait_until(
        lambda: (
            manager.get_running(session_id) is None
            or manager.get_running(session_id).current_run is None
        ),
        timeout=timeout,
    )


async def create_session(client: httpx.AsyncClient, **kwargs) -> dict:
    resp = await client.post("/api/sessions", json=kwargs or {"model": "claude-opus-4-8"})
    assert resp.status_code == 201, resp.text
    return resp.json()


def split_frames(buf: str) -> tuple[list[str], str]:
    frames: list[str] = []
    while "\n\n" in buf:
        raw, buf = buf.split("\n\n", 1)
        if raw.strip():
            frames.append(raw)
    return frames, buf


def parse_frame(raw: str) -> tuple[str, dict]:
    event_type: str | None = None
    data: dict | None = None
    for line in raw.split("\n"):
        if line.startswith("event: "):
            event_type = line[len("event: "):]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: "):])
    assert event_type is not None and data is not None, f"malformed frame: {raw!r}"
    return event_type, data


async def read_frames(
    application, path: str, count: int, timeout: float = 5.0
) -> tuple[int, dict, list[tuple[str, dict]]]:
    frames: list[tuple[str, dict]] = []
    buf = ""
    status: dict = {}
    headers: dict = {}

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal buf
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
            headers.update(
                {k.decode().lower(): v.decode() for k, v in message["headers"]}
            )
        elif message["type"] == "http.response.body":
            buf += message.get("body", b"").decode()
            ready, buf = split_frames(buf)
            for raw in ready:
                if raw.startswith(":"):
                    continue
                frames.append(parse_frame(raw))
            if len(frames) >= count:
                return

    task = asyncio.create_task(application(_scope(path), receive, send))
    try:
        await asyncio.wait_for(_wait_frames(frames, count), timeout)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return status.get("code", 0), headers, frames


async def read_raw_stream(application, path: str, duration: float) -> str:
    chunks: list[str] = []

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b"").decode())

    task = asyncio.create_task(application(_scope(path), receive, send))
    try:
        await asyncio.sleep(duration)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return "".join(chunks)


def _scope(url: str) -> dict:
    path, _, query = url.partition("?")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": url.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "server": ("test", 80),
        "client": ("test", 123),
        "headers": [(b"host", b"test"), (b"accept", b"text/event-stream")],
    }


async def _wait_frames(frames, count) -> None:
    while len(frames) < count:
        await asyncio.sleep(0.01)


async def test_health(api):
    client, _, _ = api
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "claude-opus-4-8"


async def test_create_list_get_delete_session(api):
    client, _, _ = api
    created = await create_session(client, model="m-test", effort="low", title="t1")
    assert created["id"]
    assert created["status"] == "idle"
    assert created["model"] == "m-test"
    assert created["effort"] == "low"
    assert created["title"] == "t1"
    assert created["total_input_tokens"] == 0

    listed = (await client.get("/api/sessions")).json()
    assert any(s["id"] == created["id"] for s in listed)

    detail = (await client.get(f"/api/sessions/{created['id']}")).json()
    assert detail["id"] == created["id"]

    resp = await client.delete(f"/api/sessions/{created['id']}")
    assert resp.status_code == 204
    assert (await client.get(f"/api/sessions/{created['id']}")).status_code == 404
    assert (await client.delete(f"/api/sessions/{created['id']}")).status_code == 404


async def test_post_message_returns_202_and_processes(api):
    client, application, scripted = api
    session = await create_session(client)
    scripted.turns = [end_turn("answer one")]
    resp = await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "hello"}
    )
    assert resp.status_code == 202
    assert resp.json() == {"ok": True, "session_id": session["id"], "queued": True}

    await wait_run_done(application, session["id"])
    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"][0]["text"] == "hello"
    assert messages[1]["content"][0]["text"] == "answer one"

    resp = await client.post("/api/sessions/ghost/messages", json={"text": "x"})
    assert resp.status_code == 404


async def test_get_messages_incremental_after_seq(api):
    client, application, scripted = api
    session = await create_session(client)
    scripted.turns = [end_turn("reply"), end_turn("reply2")]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "q1"})
    await wait_run_done(application, session["id"])
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "q2"})
    await wait_run_done(application, session["id"])

    all_messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    assert len(all_messages) == 4
    seqs = [m["seq"] for m in all_messages]
    assert seqs == [1, 2, 3, 4]

    incremental = (
        await client.get(f"/api/sessions/{session['id']}/messages?after_seq=2")
    ).json()
    assert [m["seq"] for m in incremental] == [3, 4]

    assert (
        await client.get("/api/sessions/ghost/messages")
    ).status_code == 404


async def test_sse_stream_wire_format_and_seq_monotonic(api):
    client, application, scripted = api
    session = await create_session(client)
    scripted.turns = [end_turn("streamed answer")]
    scripted.deltas = [[StreamDelta(kind="text", text="streamed answer")]]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "hi"})
    await wait_run_done(application, session["id"])

    status, headers, frames = await read_frames(
        application, f"/api/sessions/{session['id']}/events?after_seq=0", count=2
    )
    assert status == 200
    assert headers.get("content-type", "").startswith("text/event-stream")
    assert headers.get("cache-control") == "no-cache"
    kinds = [kind for kind, _ in frames]
    assert kinds == ["text_delta", "turn_complete"]
    seqs = [data["seq"] for _, data in frames]
    assert seqs == [1, 2]
    assert frames[0][1]["type"] == "text_delta"
    assert frames[0][1]["data"] == {"text": "streamed answer"}
    assert frames[1][1]["type"] == "turn_complete"
    assert frames[1][1]["data"]["stop_reason"] == "end_turn"
    assert frames[1][1]["data"]["usage"]["input_tokens"] == 150


async def test_sse_incremental_cursor_after_seq(api):
    client, application, scripted = api
    session = await create_session(client)
    scripted.turns = [end_turn("one"), end_turn("two")]
    scripted.deltas = [[StreamDelta(kind="text", text="one")]]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "q1"})
    await wait_run_done(application, session["id"])

    _, _, frames = await read_frames(
        application, f"/api/sessions/{session['id']}/events?after_seq=1", count=1
    )
    assert frames[0][0] == "turn_complete"
    assert frames[0][1]["seq"] == 2

    scripted.turns = [end_turn("two")]
    scripted.deltas = [[StreamDelta(kind="text", text="two")]]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "q2"})
    await wait_run_done(application, session["id"])
    _, _, frames = await read_frames(
        application, f"/api/sessions/{session['id']}/events?after_seq=2", count=2
    )
    assert [data["seq"] for _, data in frames] == [3, 4]
    assert frames[0][1]["data"]["text"] == "two"


async def test_sse_heartbeat_ping_keeps_idle_stream_alive(api, monkeypatch):
    from app.api import events as events_module

    monkeypatch.setattr(events_module, "SSE_PING_INTERVAL_S", 0.2)
    client, application, _ = api
    session = await create_session(client)

    raw = await read_raw_stream(
        application, f"/api/sessions/{session['id']}/events?after_seq=0", duration=1.0
    )
    assert ": ping" in raw
    assert raw.count(": ping") >= 3
    assert "event:" not in raw


async def test_sse_heartbeat_interleaves_with_events(api, monkeypatch):
    from app.api import events as events_module

    monkeypatch.setattr(events_module, "SSE_PING_INTERVAL_S", 0.05)
    client, application, scripted = api
    session = await create_session(client)
    scripted.turns = [end_turn("with heartbeat")]
    scripted.deltas = [[StreamDelta(kind="text", text="with heartbeat")]]
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "hi"})
    await wait_run_done(application, session["id"])

    status, _, frames = await read_frames(
        application, f"/api/sessions/{session['id']}/events?after_seq=0", count=2
    )
    assert status == 200
    assert [data["seq"] for _, data in frames] == [1, 2]


async def seed_approval_scenario(client, application, scripted) -> str:
    session = await create_session(client)
    scripted.turns = [tool_use_turn("tu_r", "risky", {"text": "rm"}), end_turn("ok done")]
    manager = manager_of(application)
    from app.tools.registry import ToolSpec

    async def risky_handler(tool_input: dict) -> dict:
        return {"blew_up": True}

    if manager.registry.get("risky") is None:
        manager.registry.register(
        ToolSpec(
            name="risky",
            description="dangerous",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=risky_handler,
            timeout_s=5,
            dangerous=True,
            access="execute",
        )
    )
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "go"})
    return session["id"]


async def test_approval_allow_completes_flow(api):
    client, application, scripted = api
    session_id = await seed_approval_scenario(client, application, scripted)
    manager = manager_of(application)

    await wait_until(
        lambda: (
            manager.get_running(session_id) is not None
            and "tu_r" in manager.get_running(session_id).pending_approvals
        )
    )
    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/tu_r", json={"result": "allow"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "ok": True,
        "session_id": session_id,
        "tool_use_id": "tu_r",
        "result": "allow",
    }

    await wait_run_done(application, session_id)
    messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    tool_result = next(
        b for m in messages for b in m["content"] if b["type"] == "tool_result"
    )
    assert tool_result["is_error"] is False
    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/tu_r", json={"result": "allow"}
    )
    assert resp.status_code == 404


async def test_approval_deny_and_unknown_pending_404(api):
    client, application, scripted = api
    session_id = await seed_approval_scenario(client, application, scripted)
    manager = manager_of(application)

    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/tu_ghost", json={"result": "allow"}
    )
    assert resp.status_code == 404
    assert resp.json()["ok"] is False

    await wait_until(
        lambda: (
            manager.get_running(session_id) is not None
            and "tu_r" in manager.get_running(session_id).pending_approvals
        )
    )
    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/tu_r", json={"result": "deny"}
    )
    assert resp.status_code == 200
    await wait_run_done(application, session_id)

    messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    tool_result = next(
        b for m in messages for b in m["content"] if b["type"] == "tool_result"
    )
    assert tool_result["is_error"] is True
    assert messages[-1]["content"][0]["text"] == "ok done"

    resp = await client.post(
        f"/api/sessions/{session_id}/approvals/tu_x", json={"result": "maybe"}
    )
    assert resp.status_code == 422


async def test_interrupt_running_session(api):
    client, application, scripted = api
    session = await create_session(client)
    manager = manager_of(application)
    blocking = BlockingLLM()
    manager.llm = blocking

    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "hang"})
    await wait_until(lambda: blocking.started.is_set())

    resp = await client.post(f"/api/sessions/{session['id']}/interrupt")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "session_id": session["id"], "interrupted": True}
    await wait_run_done(application, session["id"])

    scripted.turns = [end_turn("after interrupt")]
    manager.llm = scripted
    resp = await client.post(
        f"/api/sessions/{session['id']}/messages", json={"text": "resume"}
    )
    assert resp.status_code == 202
    await wait_run_done(application, session["id"])
    messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
    assert messages[-1]["content"][0]["text"] == "after interrupt"

    resp = await client.post(f"/api/sessions/{session['id']}/interrupt")
    assert resp.status_code == 200
    assert resp.json()["interrupted"] is False
    assert (await client.post("/api/sessions/ghost/interrupt")).status_code == 404


async def test_delete_running_session_stops_it(api):
    client, application, scripted = api
    session = await create_session(client)
    manager = manager_of(application)
    blocking = BlockingLLM()
    manager.llm = blocking
    await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "hang"})
    await wait_until(lambda: blocking.started.is_set())

    resp = await client.delete(f"/api/sessions/{session['id']}")
    assert resp.status_code == 204
    await wait_until(lambda: manager.get_running(session["id"]) is None)
    assert (await client.get(f"/api/sessions/{session['id']}")).status_code == 404


async def test_error_shape_unified_json(api):
    client, _, _ = api
    resp = await client.get("/api/sessions/ghost")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "not_found"
    assert "ghost" in body["error"]["message"]


async def test_events_stream_unknown_session_404(api):
    client, _, _ = api
    resp = await client.get("/api/sessions/ghost/events?after_seq=0")
    assert resp.status_code == 404
    assert resp.json()["ok"] is False
