"""SSE 事件协议（严格按）。

统一信封 {seq, type, data}；seq 会话内单调递增，前端用它做幂等去重
与断线续传（after_seq 游标）。七种引擎事件类型缺一不可：

    text_delta / thinking_delta / tool_call_start / tool_call_result /
    approval_request / turn_complete / error

外加两种全局广播事件（不落库、不参与 DB 推导）：

    workspace_changed （多目录工作区）
    tools_changed （自扩展 Agent：运行时安装/卸载 MCP server）

data 字段用各 *Data 模型校验后以 dict 装入信封——信封保持开放结构，
新增事件类型只改这里，不动前端信封解析。
"""

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
    TASK_OPTIONS = "task_options"   # 规划器拆解出的执行方案选项
    TASK_PROGRESS = "task_progress"  # 长时任务分钟级进度推送


class ToolsChangedData(BaseModel):
    """event: tools_changed — 运行时工具集变更（自扩展 Agent）。

    install/uninstall 成功后由后端向所有已知会话广播；前端据此 toast
    提示"工具集已更新"。tools 为带命名空间前缀（mcp__<server>__<tool>）
    的工具名列表。
    """

    action: str   # "install" / "uninstall"
    server: str
    tools: list[str] = []


class UsageOut(BaseModel):
    """单轮用量（turn_complete.usage），列名与 messages 表对齐。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0


class TextDeltaData(BaseModel):
    """event: text_delta — 流式文本。

    role 仅在 event_bus 从 DB 推导历史事件时携带（P3 联调遗留①）：
    user 消息的 text 块 → role="user"，前端据此渲染用户气泡而非
    assistant 流式文本；实时流式增量 role=None（恒为 assistant）。
    """

    text: str
    role: str | None = None


class ThinkingDeltaData(BaseModel):
    """event: thinking_delta — 可选展示（adaptive thinking）。"""

    text: str


class ToolCallStartData(BaseModel):
    """event: tool_call_start。"""

    tool_use_id: str
    name: str
    input_preview: str


class ToolCallResultData(BaseModel):
    """event: tool_call_result。"""

    tool_use_id: str
    is_error: bool
    output_preview: str


class ApprovalRequestData(BaseModel):
    """event: approval_request — 前端弹审批框。"""

    tool_use_id: str
    name: str
    input: dict[str, Any]
    reason: str | None = None


class TurnCompleteData(BaseModel):
    """event: turn_complete — 前端收尾并更新用量角标。"""

    stop_reason: str | None = None
    usage: UsageOut = UsageOut()
    cost_usd: float = 0.0


class TaskOptionData(BaseModel):
    """一个执行方案选项（task_options.options 列表项）。"""

    label: str        # 方案名（前端卡片标题，规划器约束 ≤12 字）
    detail: str = ""  # 做法 / 范围说明（卡片副文本，可为空）


class TaskOptionsData(BaseModel):
    """event: task_options — 规划器判定 medium/complex 任务后让用户先选方向。

    语义与 approval_request 同构：前端渲染选项卡片，用户点击后
    POST /api/sessions/{id}/choices/{choice_id} {selected: <下标>}，
    dispatcher 侧（实为 SessionManager 的 PendingChoice Future）被 resolve，
    AgentLoop 注入选择注记后继续正常循环。超时未选 → 后端自动采用
    recommended 方案并在注记中标明"自动采用"。
    """

    choice_id: str
    complexity: str                  # medium / complex（simple 不会发本事件）
    options: list[TaskOptionData]
    recommended: int = 0             # 推荐方案下标（超时自动采用）
    reason: str | None = None        # 规划器给出的一句话拆解理由
    timeout_s: int = 120             # 选择倒计时提示用；超时后端自动采用推荐


class TaskProgressData(BaseModel):
    """event: task_progress — 长时任务周期性进度（不落库）。

    后端扫描定义长时任务：规划器标记 long_running，或 run 运行时长超过
    long_running_after_s 自动升级。此后 AgentLoop 的进度 ticker 每
    progress_interval_s（默认 60s）发一次，前端展示分钟级进度卡。
    """

    elapsed_s: int              # 本次 run 已运行秒数
    iterations: int             # 已完成的循环迭代步数
    last_activity: str | None = None  # 最近活动描述（"模型思考中"/"执行工具 xxx"）


class ErrorData(BaseModel):
    """event: error — 任何层失败归一化后的对外形态。"""

    code: str
    message: str


# 各事件类型 → data 载荷的映射，供 API 层 / event_bus 构造事件时校验
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
    """统一信封：{seq, type, data}。"""

    seq: int
    type: SSEEventType
    data: dict[str, Any]


def make_event(
    seq: int,
    event_type: SSEEventType,
    data: BaseModel | Mapping[str, Any],
) -> SSEEvent:
    """构造信封事件。传入 BaseModel 时先校验并 dump 为 dict。

    exclude_none：可选字段（如 TextDeltaData.role / ApprovalRequestData.reason）
    为 None 时不下发——实时流 wire format 与历史抓包保持一致，仅在确有值时
    增加键（推导历史事件的 user text 带 role="user"）。
    """
    payload = (
        data.model_dump(exclude_none=True) if isinstance(data, BaseModel) else dict(data)
    )
    return SSEEvent(seq=seq, type=event_type, data=payload)
