
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):

    model: Optional[str] = None
    effort: Optional[str] = None
    title: Optional[str] = None
    user_id: Optional[str] = None


class SessionOut(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    status: str
    model: str
    effort: str
    total_input_tokens: int
    total_output_tokens: int
    cache_read_tokens: int
    total_cost_usd: float
    created_at: datetime
    updated_at: datetime
