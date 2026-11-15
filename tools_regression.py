""" （协议级，对真实 mock 服务器执行）。

用法：
    python tools_regression.py full # 全链路 + 断线重连 + 刷新恢复 + 抽屉操作
    python tools_regression.py after-restart <old_last_seq> # 重启后重同步信号验证

每项输出 PASS/FAIL，退出码非 0 表示有 FAIL。
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"

results: list[tuple[str, str, str]] = []  # (item, PASS/FAIL, detail)


def report(item: str, ok: bool, detail: str = "") -> None:
    results.append((item, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {item}" + (f" — {detail}" if detail else ""))


def frame_kind(block: str) -> str:
    return block.splitlines()[0][len("event: "):]


def frame_data(block: str) -> dict:
    line = next(l for l in block.splitlines() if l.startswith("data: "))
    return json.loads(line[len("data: "):])


class SSECollector:
    """可控断开的 SSE 收集器（模拟真实前端 fetch 流）。"""

    def __init__(self, client: httpx.AsyncClient, url: str):
        self.frames: list[str] = []
        self.client = client
        self.url = url
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> list[str]:
        raw = ""
        async with self.client.stream("GET", self.url, headers={"Accept": "text/event-stream"}) as resp:
            async for chunk in resp.aiter_text():
                raw += chunk.replace("\r\n", "\n")
                while "\n\n" in raw:
                    block, raw = raw.split("\n\n", 1)
                    if block.strip():
                        self.frames.append(block)
                        if block.startswith("event: turn_complete"):
                            return self.frames
        return self.frames

    def kill(self) -> None:
        """硬断 SSE 连接（模拟网络断开/杀连接）。"""
        if self.task and not self.task.done():
            self.task.cancel()

    async def wait_done(self, timeout: float = 15.0) -> list[str]:
        return await asyncio.wait_for(self.task, timeout)

    @property
    def last_seq(self) -> int:
        return frame_data(self.frames[-1])["seq"] if self.frames else 0


def seqs_of(frames: list[str]) -> list[int]:
    return [frame_data(f)["seq"] for f in frames]


async def full(client: httpx.AsyncClient) -> int:
    # ---- ① 发消息 → 流式渲染 → 收尾 全链路 ----
    sid = (await client.post("/api/sessions", json={"title": "回归-全链路"})).json()["id"]
    collector = SSECollector(client, f"/api/sessions/{sid}/events?after_seq=0")
    collector.start()
    await asyncio.sleep(0.3)
    ack = await client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
    frames = await collector.wait_done()
    kinds = [frame_kind(f) for f in frames]
    seqs = seqs_of(frames)
    report(
        "① 发消息→流式→收尾全链路",
        ack.status_code == 202
        and kinds[0] in ("thinking_delta", "text_delta")
        and kinds[-1] == "turn_complete"
        and seqs == list(range(1, len(seqs) + 1)),
        f"kinds头尾={kinds[:1]}..{kinds[-1:]} seq连续1..{len(seqs)}",
    )
    tc = frame_data(frames[-1])

    def mock_cost_usd(u: dict) -> float:
        """ 后 turn_complete.cost_usd 是定价表真实计算值（claude-opus-4-8：
        输入 $5 / 输出 $25 per MTok，cache 读 0.1x、写 1.25x）。mock 服务器
        用真实引擎发 turn_complete，故这里按同口径复算并对账。"""
        rate = {"in": 5.0, "out": 25.0, "cr": 0.5, "cw": 6.25}
        return round(
            (
                u.get("input_tokens", 0) * rate["in"]
                + u.get("output_tokens", 0) * rate["out"]
                + u.get("cache_read", 0) * rate["cr"]
                + u.get("cache_creation", 0) * rate["cw"]
            ) / 1_000_000,
            6,
        )

    expected_cost = mock_cost_usd(tc["data"]["usage"])
    report(
        "①b turn_complete 载荷字段（cost_usd 定价表真实值）",
        tc["data"]["usage"].keys() >= {"input_tokens", "output_tokens", "cache_read"}
        and expected_cost > 0
        and abs(tc["data"]["cost_usd"] - expected_cost) < 1e-9,
        f"usage={tc['data']['usage']} cost={tc['data']['cost_usd']} "
        f"expected={expected_cost}",
    )

    # ---- ② 杀 SSE 连接 → after_seq 续传不丢不重 ----
    sid2 = (await client.post("/api/sessions", json={})).json()["id"]
    c2 = SSECollector(client, f"/api/sessions/{sid2}/events?after_seq=0")
    c2.start()
    await asyncio.sleep(0.3)
    await client.post(f"/api/sessions/{sid2}/messages", json={"text": "slow 慢速输出"})
    await asyncio.sleep(1.2)  # slow 场景 delta 间隔 0.25s → 收到部分帧
    killed_at = c2.last_seq
    c2.kill()  # 硬断连接（此时 turn 尚未完成）
    await asyncio.sleep(0.3)
    # 前端语义：指数退避后带 after_seq=lastSeq 重连
    c2b = SSECollector(client, f"/api/sessions/{sid2}/events?after_seq={killed_at}")
    c2b.start()
    rest = await c2b.wait_done()
    rest_seqs = seqs_of(rest)
    expect_first = killed_at + 1
    ok = (
        bool(rest_seqs)
        and rest_seqs[0] == expect_first
        and rest_seqs == list(range(expect_first, expect_first + len(rest_seqs)))
        and rest[-1].startswith("event: turn_complete")
    )
    report(
        "② 断线重连 after_seq 续传",
        ok,
        f"杀连接时seq={killed_at} → 重连首帧seq={rest_seqs[0] if rest_seqs else '?'} "
        f"续传{len(rest)}帧到turn_complete 无重复",
    )

    # ---- ③ 刷新页面 → 历史恢复（messages 全量 + 事件全量重放双通道）----
    msgs = (await client.get(f"/api/sessions/{sid2}/messages")).json()
    replay_collector = SSECollector(client, f"/api/sessions/{sid2}/events?after_seq=0")
    replay_collector.start()
    replay = await replay_collector.wait_done()
    msg_seqs = [m["seq"] for m in msgs]
    replay_seqs = seqs_of(replay)
    ok = (
        len(msgs) >= 2
        and msgs[0]["role"] == "user"
        and msgs[-1]["role"] == "assistant"
        and msgs[-1]["content"][0]["text"].startswith("这")  # slow 场景正文完整落库
        and replay_seqs == list(range(1, len(replay_seqs) + 1))
        and replay[-1].startswith("event: turn_complete")
        and max(msg_seqs) < max(replay_seqs)  # 两个独立 seq 空间（前端不得混用）
    )
    report(
        "③ 刷新页面历史恢复",
        ok,
        f"messages.seq={msg_seqs} 事件重放seq=1..{max(replay_seqs)} "
        f"(独立空间: max(msg)={max(msg_seqs)} < max(evt)={max(replay_seqs)})",
    )

    # ---- ④ 会话抽屉：列表 / 新建 / 切换 / 删除 ----
    before = (await client.get("/api/sessions")).json()
    created = (await client.post("/api/sessions", json={"title": "回归-抽屉"})).json()
    after = (await client.get("/api/sessions")).json()
    # 切换到旧会话再回来：旧会话事件重放可用（切换 = openSession + 全量重放）
    switch_back = SSECollector(client, f"/api/sessions/{sid}/events?after_seq=0")
    switch_back.start()
    switch_frames = await switch_back.wait_done()
    deleted = await client.delete(f"/api/sessions/{created['id']}")
    gone = await client.get(f"/api/sessions/{created['id']}")
    final = (await client.get("/api/sessions")).json()
    ok = (
        len(after) == len(before) + 1
        and len(switch_frames) == len(frames)  # 切回旧会话重放出同样多的事件
        and deleted.status_code == 204
        and gone.status_code == 404
        and len(final) == len(before)
    )
    report(
        "④ 会话抽屉 列表/新建/切换/删除",
        ok,
        f"新建前后列表 {len(before)}→{len(after)}；切回旧会话重放 {len(switch_frames)} 帧；"
        f"删除→204→详情404；终态列表 {len(final)}",
    )

    # 返回游标供 after-restart 阶段用
    print(f"\nRESTART_CURSOR={c2.last_seq}")
    print(f"RESTART_SESSION={sid2}")
    return 0


async def after_restart(client: httpx.AsyncClient, sid: str, old_cursor: int) -> int:
    """服务重启后：旧游标超出新进程 seq 空间 → 服务端全量重同步重放（seq 从 1 起）。

    这就是前端「seq ≤ 连接基线 → 整体替换本地状态」信号的来源。
    """
    c = SSECollector(client, f"/api/sessions/{sid}/events?after_seq={old_cursor}")
    c.start()
    frames = await c.wait_done()
    seqs = seqs_of(frames)
    ok = bool(seqs) and seqs[0] == 1 and seqs == list(range(1, len(seqs) + 1))
    report(
        "⑤ 服务重启 → 全量重同步信号（P1-3 遗留①）",
        ok,
        f"旧游标={old_cursor} → 重放首帧seq={seqs[0] if seqs else '?'}（应为1）"
        f" 共{len(frames)}帧 含turn_complete={frames[-1].startswith('event: turn_complete')}",
    )
    return 0


async def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    async with httpx.AsyncClient(base_url=BASE, timeout=20.0) as client:
        if mode == "full":
            code = await full(client)
        elif mode == "after-restart":
            sid, cursor = sys.argv[2], int(sys.argv[3])
            code = await after_restart(client, sid, cursor)
        else:
            raise SystemExit(f"unknown mode {mode}")
    fails = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(fails)}/{len(results)} PASS, {len(fails)} FAIL")
    return 1 if fails else code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
