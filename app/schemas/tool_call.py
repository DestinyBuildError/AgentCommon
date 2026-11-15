
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    message_id: str
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    result: Optional[Any] = None
    is_error: bool
    status: str
    approval: str
    duration_ms: Optional[int] = None
    created_at: datetime
