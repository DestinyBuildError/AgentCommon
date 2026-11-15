
from __future__ import annotations

import datetime

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="demo", instructions="Demo server: time & echo utilities.")


@mcp.tool()
def current_time() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@mcp.tool()
def echo_text(text: str) -> str:
    return text


if __name__ == "__main__":
    mcp.run()
