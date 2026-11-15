"""Demo MCP server（stdio 传输）—— 开箱演示用。

提供 2 个极简工具，让项目无需任何第三方依赖即可演示 MCP 全链路
（config.yaml.example 已预置本 server）：
  - current_time() 返回当前本地时间（ISO-8601）
  - echo_text(text) 原样回显文本

启动方式（由 MCPManager 以子进程 stdio 拉起，不手动运行）：
  config.yaml:
    mcp_servers:
      - name: demo
        command: "python"
        args: ["demo_mcp_server.py"]

协议：MCP stdio（JSON-RPC over stdin/stdout），基于官方 mcp SDK 的
MCPServer（2.x；1.x 时代的 FastMCP 在 2.x 更名为 MCPServer）。
注意：stdio 传输独占 stdout——不要在这里 print() 调试信息。
"""

from __future__ import annotations

import datetime

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="demo", instructions="Demo server: time & echo utilities.")


@mcp.tool()
def current_time() -> str:
    """Return the current local time in ISO-8601 format (e.g. 2026-01-12T08:30:00)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


@mcp.tool()
def echo_text(text: str) -> str:
    """Echo the given text back to the caller, verbatim."""
    return text


if __name__ == "__main__":
    mcp.run()  # stdio transport（默认）
