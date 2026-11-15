
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ApprovalDecision(BaseModel):

    result: Literal["allow", "deny"]
