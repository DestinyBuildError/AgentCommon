"""工具注册表（tools/、缓存纪律）。

- ToolSpec：工具静态描述 + 执行句柄 + 调度策略标记
- ToolRegistry：register / list / get；列表按 name 确定性排序
  （缓存纪律：工具列表顺序确定 → prompt caching 前缀稳定）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# 工具访问类别（权限控制）：dispatcher 按 (权限模式, access) 矩阵
# 决定 直接执行 / 走审批门 / 拒绝。见 dispatcher.PERMISSION_MATRIX。
ACCESS_LEVELS = ("read", "write", "execute", "admin")


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的注册描述。

    handler: async (input: dict) -> Any
      成功返回任意可 JSON 化结果；失败抛 ToolError 子类（dispatcher 归一化）。
    timeout_s: 每次执行的超时上限（dispatcher 用 asyncio.wait_for 强杀）
    parallel_safe: True → 与其他工具并发执行；False → dispatcher 串行化
    dangerous: True → 语义上属危险操作（审批门全开模式外的放行由 access 矩阵定）
    access: 访问类别（read 只读 / write 文件写删与目录管理 / execute 命令与
      MCP 调用 / admin 元操作如安装卸载 MCP server）——权限矩阵的判据，
      未知值按最保守的 approval 处理
    retryable: 瞬时失败（ToolTransientError）的内部重试次数，默认 0（第②层）
    """

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
        """Anthropic tools 入参形态。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """进程内工具注册表。register 时不排序，list/api_schemas 时确定性排序。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> bool:
        """注销工具（运行时卸载 MCP server 时移除其工具）。

        不存在返回 False（幂等友好）；存在即移除——api_schemas() 每轮迭代
        实时重算，下一迭代模型就看不到该工具了。
        """
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        """确定性排序的工具名列表（未知工具报错时引用）。"""
        return sorted(self._tools)

    def list(self) -> list[ToolSpec]:
        """按 name 排序的全部工具（确定性 → prompt cache 前缀稳定）。"""
        return [self._tools[name] for name in sorted(self._tools)]

    def api_schemas(self) -> list[dict[str, Any]]:
        """发给 Anthropic API 的 tools 参数（排序确定）。"""
        return [spec.api_schema() for spec in self.list()]

    def __len__(self) -> int:
        return len(self._tools)
