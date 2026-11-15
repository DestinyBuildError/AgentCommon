
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel

PermissionMode = Literal["read", "write", "full"]


class PermissionUpdate(BaseModel):

    mode: PermissionMode


class PermissionOut(BaseModel):

    mode: PermissionMode
