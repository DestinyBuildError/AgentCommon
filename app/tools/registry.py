
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ACCESS_LEVELS = ("read", "write", "execute", "admin")


@dataclass(frozen=True)
class ToolSpec:

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    timeout_s: float = 60.0
    parallel_safe: bool = True
    dangerous: bool = False
    access: str = "read"
    retryable: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def api_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]

    def api_schemas(self) -> list[dict[str, Any]]:
        return [spec.api_schema() for spec in self.list()]

    def __len__(self) -> int:
        return len(self._tools)
