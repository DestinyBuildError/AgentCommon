
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="text_tools", instructions="Text utilities: reverse / case / count.")


@mcp.tool()
def reverse_text(text: str) -> str:
    return text[::-1]


@mcp.tool()
def to_uppercase(text: str) -> str:
    return text.upper()


@mcp.tool()
def count_characters(text: str) -> str:
    return str(len(text))


if __name__ == "__main__":
    mcp.run()
