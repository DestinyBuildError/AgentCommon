"""目录 demo server：text_tools（stdio 传输）—— 文本处理工具集。

由 discover_tools 在工具目录中检索、install_mcp_server 以子进程 stdio 拉起
（见 tool_catalog.yaml）。提供 3 个纯函数工具：
  - reverse_text(text) 反转字符串
  - to_uppercase(text) 转大写
  - count_characters(text) 返回字符数

协议与 demo_mcp_server.py 相同：MCP stdio（JSON-RPC over
stdin/stdout）。注意：stdio 传输独占 stdout——不要在这里 print() 调试信息。
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="text_tools", instructions="Text utilities: reverse / case / count.")


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse the given text character by character (e.g. 'abc' -> 'cba')."""
    return text[::-1]


@mcp.tool()
def to_uppercase(text: str) -> str:
    """Convert the given text to upper case."""
    return text.upper()


@mcp.tool()
def count_characters(text: str) -> str:
    """Count the number of characters in the given text."""
    return str(len(text))


if __name__ == "__main__":
    mcp.run()  # stdio transport（默认）
