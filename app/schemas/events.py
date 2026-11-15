
from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Union

from pydantic import BaseModel

from app.schemas.workspace import WorkspaceChangedData


class SSEEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    APPROVAL_REQUEST = "approval_request"
    TURN_COMPLETE = "turn_complete"
    ERROR = "error"
    WORKSPACE_CHANGED = "workspace_changed"
    TOOLS_CHANGED = "tools_changed"
    TASK_OPTIONS = "task_options"
    TASK_PROGRESS = "task_progress"


class ToolsChangedData(BaseModel):

    action: str
    server: str
    tools: list[str] = []


class UsageOut(BaseModel):

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0


class TextDeltaData(BaseModel):

    text: str
    role: str | None = None


class ThinkingDeltaData(BaseModel):

    text: str


class ToolCallStartData(BaseModel):

    tool_use_id: str
    name: str
    input_preview: str


class ToolCallResultData(BaseModel):

    tool_use_id: str
    is_error: bool
    output_preview: str


class ApprovalRequestData(BaseModel):

    tool_use_id: str
    name: str
    input: dict[str, Any]
    reason: str | None = None


class TurnCompleteData(BaseModel):

    stop_reason: str | None = None
    usage: UsageOut = UsageOut()
    cost_usd: float = 0.0


class TaskOptionData(BaseModel):

    label: str
    detail: str = ""


class TaskOptionsData(BaseModel):

    choice_id: str
    complexity: str
    options: list[TaskOptionData]
    recommended: int = 0
    reason: str | None = None
    timeout_s: int = 120


class TaskProgressData(BaseModel):

    elapsed_s: int
    iterations: int
    last_activity: str | None = None


class ErrorData(BaseModel):

    code: str
    message: str


EVENT_DATA_TYPES: dict[SSEEventType, type[BaseModel]] = {
    SSEEventType.TEXT_DELTA: TextDeltaData,
    SSEEventType.THINKING_DELTA: ThinkingDeltaData,
    SSEEventType.TOOL_CALL_START: ToolCallStartData,
    SSEEventType.TOOL_CALL_RESULT: ToolCallResultData,
    SSEEventType.APPROVAL_REQUEST: ApprovalRequestData,
    SSEEventType.TURN_COMPLETE: TurnCompleteData,
    SSEEventType.ERROR: ErrorData,
    SSEEventType.WORKSPACE_CHANGED: WorkspaceChangedData,
    SSEEventType.TOOLS_CHANGED: ToolsChangedData,
    SSEEventType.TASK_OPTIONS: TaskOptionsData,
    SSEEventType.TASK_PROGRESS: TaskProgressData,
}

EventData = Union[
    TextDeltaData,
    ThinkingDeltaData,
    ToolCallStartData,
    ToolCallResultData,
    ApprovalRequestData,
    TurnCompleteData,
    ErrorData,
    WorkspaceChangedData,
    ToolsChangedData,
    TaskOptionsData,
    TaskProgressData,
]


class SSEEvent(BaseModel):

    seq: int
    type: SSEEventType
    data: dict[str, Any]


def make_event(
    seq: int,
    event_type: SSEEventType,
    data: BaseModel | Mapping[str, Any],
) -> SSEEvent:
    payload = (
        data.model_dump(exclude_none=True) if isinstance(data, BaseModel) else dict(data)
    )
    return SSEEvent(seq=seq, type=event_type, data=payload)
