"""消息 API 模型（：GET /api/sessions/{id}/messages?after_seq=N）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class MessageCreate(BaseModel):
    """发消息请求体（：POST /api/sessions/{id}/messages）。"""

    text: str


class MessageOut(BaseModel):
    """一条对话记录。content 是完整 content blocks（含 tool_use / tool_result），
    不是纯文本（：API 重放和 repair_orphans 的前提）。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    seq: int
    role: str  # user / assistant
    content: list[dict[str, Any]]
    stop_reason: Optional[str] = None
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_creation: int
    created_at: datetime
