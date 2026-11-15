
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class WorkspaceDirOut(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: str
    path: str
    source: Literal["user", "agent"]
    created_at: datetime


class WorkspaceListOut(BaseModel):

    ok: bool = True
    dirs: list[WorkspaceDirOut]


class WorkspaceCreateIn(BaseModel):

    path: str


class WorkspaceDirCreatedOut(BaseModel):

    ok: bool = True
    dir: WorkspaceDirOut


class WorkspaceChangedData(BaseModel):

    action: Literal["add", "remove"]
    path: str
    source: Literal["user", "agent"]
    dirs: list[WorkspaceDirOut]
