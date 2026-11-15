"""Pydantic v2 API 模型 —— 前后端契约（/）。

从本包根直接导入，作为下一波次的稳定接口面：
    from app.schemas import SessionCreate, SessionOut, MessageOut, ToolCallOut, SSEEvent
"""

from app.schemas.events import (
    ApprovalRequestData,
    ErrorData,
    SSEEvent,
    SSEEventType,
    TextDeltaData,
    ThinkingDeltaData,
    ToolCallResultData,
    ToolCallStartData,
    TurnCompleteData,
    UsageOut,
    make_event,
)
from app.schemas.message import MessageOut
from app.schemas.session import SessionCreate, SessionOut
from app.schemas.tool_call import ToolCallOut

__all__ = [
    # sessions
    "SessionCreate",
    "SessionOut",
    # messages
    "MessageOut",
    # tool_calls
    "ToolCallOut",
    # SSE 事件协议
    "SSEEvent",
    "SSEEventType",
    "UsageOut",
    "TextDeltaData",
    "ThinkingDeltaData",
    "ToolCallStartData",
    "ToolCallResultData",
    "ApprovalRequestData",
    "TurnCompleteData",
    "ErrorData",
    "make_event",
]
