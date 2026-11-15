"""工具异常类型 —— 失败分类学的工具侧表达。

工具 handler 通过抛出这三类异常表达失败语义；其余异常由 dispatcher
兜底归一化为 internal_error。结构化错误文本由 dispatcher.tool_error 生成。
"""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """工具失败基类。code 会成为 tool_error JSON 的 error.type。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_error",
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class ToolInputError(ToolError):
    """参数不合法：模型传错类型 / 缺必填字段 → is_error + schema 提示，模型自纠。"""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="invalid_params", hint=hint)


class ToolTransientError(ToolError):
    """瞬时失败（网络抖动等）：按 spec.retryable 内部重试，仍失败才回 error。"""

    def __init__(self, message: str, *, code: str = "transient_failure", hint: str | None = None) -> None:
        super().__init__(message, code=code, hint=hint)


class ToolPermanentError(ToolError):
    """永久失败（文件不存在 / 无权限）：立即回 error，附可执行建议。"""

    def __init__(self, message: str, *, code: str = "permanent_failure", hint: str | None = None) -> None:
        super().__init__(message, code=code, hint=hint)


def tool_error(code: str, message: str, hint: str | None = None) -> str:
    """结构化错误文本（错误内容格式）——tool_result(is_error=True) 的 content。

    例：tool_error("not_found", f"File not found: {path}",
                    hint=f"Directory contains: {sorted(os.listdir(parent))[:10]}")
    """
    import json

    payload: dict[str, Any] = {"error": {"type": code, "message": message}}
    if hint:
        payload["error"]["hint"] = hint
    return json.dumps(payload, ensure_ascii=False)
