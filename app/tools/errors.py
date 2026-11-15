
from __future__ import annotations

from typing import Any


class ToolError(Exception):

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

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="invalid_params", hint=hint)


class ToolTransientError(ToolError):

    def __init__(self, message: str, *, code: str = "transient_failure", hint: str | None = None) -> None:
        super().__init__(message, code=code, hint=hint)


class ToolPermanentError(ToolError):

    def __init__(self, message: str, *, code: str = "permanent_failure", hint: str | None = None) -> None:
        super().__init__(message, code=code, hint=hint)


def tool_error(code: str, message: str, hint: str | None = None) -> str:
    import json

    payload: dict[str, Any] = {"error": {"type": code, "message": message}}
    if hint:
        payload["error"]["hint"] = hint
    return json.dumps(payload, ensure_ascii=False)
