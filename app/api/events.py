
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

SSE_PING_INTERVAL_S = 15.0


def _sse_frame(event: SSEEvent) -> str:
    return f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"


@router.get("/sessions/{session_id}/events")
async def stream_events(
    session_id: str,
    after_seq: int = 0,
    db: AsyncSession = Depends(get_db),
    manager: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    session = await Repository(db).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")

    async def event_source() -> AsyncIterator[str]:
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
                    yield ": ping\n\n"
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
