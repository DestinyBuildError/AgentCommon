
from __future__ import annotations

from pydantic import BaseModel, field_validator


class ChoiceDecision(BaseModel):

    selected: int

    @field_validator("selected")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("selected must be >= 0 (options index)")
        return value
