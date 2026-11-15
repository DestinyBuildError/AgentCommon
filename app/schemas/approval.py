"""审批 API 模型（：POST /api/sessions/{id}/approvals/{tool_use_id}）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ApprovalDecision(BaseModel):
    """审批结果：allow / deny（超时自动 deny 由服务端处理，前端只传显式决议）。"""

    result: Literal["allow", "deny"]
