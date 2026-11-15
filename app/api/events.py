"""SSE 事件流端点（协议与断线重连）。

wire 格式：event: <type> 行 + data: <JSON> 行 + 空行分隔；
data 为统一信封 {"seq":..,"type":..,"data":{..}}。客户端断开时
StreamingResponse 关闭生成器，订阅在 finally 中注销，优雅结束。

心跳（真实模式缺陷 #2 治本项）：空闲超过 SSE_PING_INTERVAL_S 秒发一条
SSE 注释帧 `: ping\\n\\n` 保活——vite dev proxy / Nginx 等中间代理会掐
空闲连接（实测 ECONNRESET → 前端重连风暴）。注释帧是 SSE 规范的一部分，
EventSource 与 fetch 解析器都按规范忽略；前端流式看门狗也把 ping 当
存活字节，不会再误判半开连接。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session_manager
from app.core.session_manager import SessionManager
from app.db.database import get_db
from app.db.repository import Repository
from app.schemas.events import SSEEvent
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["events"])

# 心跳间隔（秒）。空闲超时即发 `: ping` 注释帧。测试可 monkeypatch 本常量
# 缩短间隔；生产 15s 足以穿过常见代理的空闲超时（一般 30~60s）。
SSE_PING_INTERVAL_S = 15.0


def _sse_frame(event: SSEEvent) -> str:
    """一条事件 → SSE wire 帧（event:/data: 行 + 空行分隔）。"""
    return f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"


@router.get("/sessions/{session_id}/events")
async def stream_events(
    session_id: str,
    after_seq: int = 0,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """SSE 事件流。after_seq=N：先重放 N 之后的事件（缓冲精确重放；缓冲
    覆盖不到时从 DB 推导历史补齐），再接实时流（断线重连协议）。"""
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")

    async def event_source() -> AsyncIterator[str]:
        # 事件泵：订阅总线 → 常规任务写入 out 队列。心跳不能直接在订阅
        # 生成器上 wait_for(anext(...))——取消 __anext__ 会把 CancelledError
        # 注入生成器帧，反复取消有异步生成器状态损坏风险；普通任务可安全
        # cancel/gather。
        out: asyncio.Queue = asyncio.Queue()
        after = max(after_seq, 0)

        async def pump() -> None:
            publisher = manager.bus.subscribe(session_id, after_seq=after)
            async with aclosing(publisher):
                async for event in publisher:
                    await out.put(event)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        out.get(), timeout=SSE_PING_INTERVAL_S
                    )
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # SSE 注释帧：保活，客户端按规范忽略
                    continue
                yield _sse_frame(event)
        finally:
            pump_task.cancel()
            await asyncio.gather(pump_task, return_exceptions=True)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
