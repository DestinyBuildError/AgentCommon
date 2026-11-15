
from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

BASE = "http://127.0.0.1:8000"
OUT_DIR = os.path.join(os.path.dirname(__file__), "contract_capture")


def save(name: str, payload) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {path} ({len(text)} bytes)")


def frame_kind(block: str) -> str:
    return block.splitlines()[0][len("event: "):]


def frame_data(block: str) -> dict:
    line = next(l for l in block.splitlines() if l.startswith("data: "))
    return json.loads(line[len("data: "):])


class SSECollector:

    def __init__(self, client: httpx.AsyncClient, url: str, max_frames: int = 500):
        self.frames: list[str] = []
        self.client = client
        self.url = url
        self.max_frames = max_frames
        self.headers_saved = False
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> list[str]:
        raw = ""
        async with self.client.stream(
            "GET", self.url, headers={"Accept": "text/event-stream"}
        ) as resp:
            if not self.headers_saved:
                save(
                    "sse_headers.txt",
                    f"status={resp.status_code}\n"
                    + "\n".join(f"{k}: {v}" for k, v in resp.headers.items()),
                )
                self.headers_saved = True
            async for chunk in resp.aiter_text():
                raw += chunk.replace("\r\n", "\n")
                while "\n\n" in raw:
                    block, raw = raw.split("\n\n", 1)
                    if block.strip():
                        self.frames.append(block)
                        if block.startswith("event: turn_complete"):
                            return self.frames
                        if len(self.frames) >= self.max_frames:
                            return self.frames
        return self.frames

    async def wait_for(self, pred, timeout: float = 10.0) -> bool:
        async with asyncio.timeout(timeout):
            while not pred(self.frames):
                await asyncio.sleep(0.05)
        return True

    async def wait_done(self, timeout: float = 15.0) -> list[str]:
        return await asyncio.wait_for(self.task, timeout)


async def run_turn(
    client: httpx.AsyncClient,
    sid: str,
    text: str,
    decision: str | None = None,
    scenario_label: str = "",
) -> list[str]:
    collector = SSECollector(client, f"/api/sessions/{sid}/events?after_seq=0")
    collector.start()
    await asyncio.sleep(0.3)
    resp = await client.post(f"/api/sessions/{sid}/messages", json={"text": text})
    assert resp.status_code == 202, resp.text

    if decision is not None:
        await collector.wait_for(
            lambda fs: any(f.startswith("event: approval_request") for f in fs)
        )
        block = next(f for f in collector.frames if f.startswith("event: approval_request"))
        tool_use_id = frame_data(block)["data"]["tool_use_id"]
        r = await client.post(
            f"/api/sessions/{sid}/approvals/{tool_use_id}", json={"result": decision}
        )
        print(f"  POST /approvals {decision} -> {r.status_code} {r.json()}")

    frames = await collector.wait_done()
    if scenario_label:
        save(scenario_label, "\n\n".join(frames) + "\n\n")
        print(f"  frame kinds: {[frame_kind(f) for f in frames]}")
    return frames


async def main() -> int:
    findings: list[str] = []
    async with httpx.AsyncClient(base_url=BASE, timeout=15.0) as client:
        r = await client.get("/health")
        save("01_health.json", r.json())
        r = await client.get("/api/sessions/ghost")
        findings.append(f"GET 404 body keys={sorted(r.json())} -> {r.json()}")
        save("02_error_shape_404.json", r.json())

        r = await client.post("/api/sessions", json={"title": "联调抓包会话"})
        assert r.status_code == 201, r.text
        session = r.json()
        sid = session["id"]
        save("03_session_create.json", session)
        findings.append("POST /sessions 201 keys=" + ",".join(session))

        save("04_session_list.json", (await client.get("/api/sessions")).json())
        save("05_session_detail.json", (await client.get(f"/api/sessions/{sid}")).json())

        print("== chat scenario ==")
        ack = await client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
        assert ack.status_code == 202
        findings.append(f"POST /messages {ack.status_code} body={ack.json()}")
        save("06_message_ack.json", {"status": ack.status_code, "body": ack.json()})
        collector = SSECollector(client, f"/api/sessions/{sid}/events?after_seq=0")
        collector.start()
        frames = await collector.wait_done()
        save("07_sse_chat_frames.txt", "\n\n".join(frames) + "\n\n")
        findings.append(f"SSE chat frame kinds={[frame_kind(f) for f in frames]}")

        print("== file scenario ==")
        sid2 = (await client.post("/api/sessions", json={})).json()["id"]
        frames2 = await run_turn(client, sid2, "读一下文件", scenario_label="08_sse_tool_frames.txt")
        findings.append(f"SSE file frame kinds={[frame_kind(f) for f in frames2]}")

        print("== approval allow scenario ==")
        sid3 = (await client.post("/api/sessions", json={})).json()["id"]
        frames3 = await run_turn(
            client, sid3, "帮我部署", decision="allow", scenario_label="11_sse_approval_allow.txt"
        )
        findings.append(f"SSE approval-allow kinds={[frame_kind(f) for f in frames3]}")

        print("== approval deny scenario ==")
        sid4 = (await client.post("/api/sessions", json={})).json()["id"]
        frames4 = await run_turn(
            client, sid4, "帮我部署", decision="deny", scenario_label="12_sse_approval_deny.txt"
        )
        findings.append(f"SSE approval-deny kinds={[frame_kind(f) for f in frames4]}")

        all_msgs = (await client.get(f"/api/sessions/{sid2}/messages")).json()
        save("13_messages_all.json", all_msgs)
        save("14_messages_after_seq.json", (await client.get(f"/api/sessions/{sid2}/messages?after_seq=2")).json())
        save("15_session_detail_after_usage.json", (await client.get(f"/api/sessions/{sid2}")).json())
        findings.append(f"GET messages seqs={[m['seq'] for m in all_msgs]}")

        last = frame_data(frames2[-2])["seq"]
        inc = await read_incremental(client, sid2, after_seq=last)
        findings.append(f"SSE after_seq={last} replayed kinds={[frame_kind(f) for f in inc]}")
        save("16_sse_incremental_replay.txt", "\n\n".join(inc) + "\n\n")

        r = await client.post(f"/api/sessions/{sid4}/interrupt")
        findings.append(f"POST /interrupt {r.status_code} body={r.json()}")
        save("17_interrupt_ack.json", {"status": r.status_code, "body": r.json()})
        r = await client.delete(f"/api/sessions/{sid4}")
        findings.append(f"DELETE /sessions -> {r.status_code}")

    save("00_findings.txt", "\n".join(findings) + "\n")
    print("\n".join(findings))
    return 0


async def read_incremental(client: httpx.AsyncClient, sid: str, after_seq: int) -> list[str]:
    collector = SSECollector(client, f"/api/sessions/{sid}/events?after_seq={after_seq}")
    collector.start()
    try:
        return await asyncio.wait_for(collector.task, timeout=2.0)
    except (asyncio.TimeoutError, TimeoutError):
        collector.task.cancel()
        return collector.frames


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
