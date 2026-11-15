
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class MessageCreate(BaseModel):

    text: str


class MessageOut(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    seq: int
    role: str
    content: list[dict[str, Any]]
    stop_reason: Optional[str] = None
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_creation: int
    created_at: datetime
