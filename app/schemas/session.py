"""会话 API 模型（：POST/GET /api/sessions 契约）。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    """创建会话请求体。model / effort 缺省时由 API 层回落到 Settings 默认值。"""

    model: Optional[str] = None
    effort: Optional[str] = None
    title: Optional[str] = None
    user_id: Optional[str] = None


class SessionOut(BaseModel):
    """会话详情 / 列表项，含累计用量（：每轮累加，列表页直接展示）。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    status: str  # idle / running / terminated
    model: str
    effort: str
    total_input_tokens: int
    total_output_tokens: int
    cache_read_tokens: int
    total_cost_usd: float
    created_at: datetime
    updated_at: datetime
