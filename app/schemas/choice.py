"""任务拆解选择契约：POST /choices 请求体。"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class ChoiceDecision(BaseModel):
    """用户对 task_options 的选择：selected 为 options 列表下标（0 起）。"""

    selected: int

    @field_validator("selected")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("selected must be >= 0 (options index)")
        return value
