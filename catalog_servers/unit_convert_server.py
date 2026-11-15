"""目录 demo server：unit_convert（stdio 传输）—— 单位换算工具集。

由 discover_tools 在工具目录中检索、install_mcp_server 以子进程 stdio 拉起
（见 tool_catalog.yaml）。提供 4 个换算工具：
  - celsius_to_fahrenheit(celsius) 摄氏 → 华氏
  - fahrenheit_to_celsius(fahrenheit) 华氏 → 摄氏
  - kilometers_to_miles(km) 公里 → 英里
  - miles_to_kilometers(miles) 英里 → 公里

协议与 demo_mcp_server.py 相同：MCP stdio（JSON-RPC over
stdin/stdout）。注意：stdio 传输独占 stdout——不要在这里 print() 调试信息。
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="unit_convert", instructions="Unit conversions: temperature & length.")


@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit (F = C * 9/5 + 32)."""
    return f"{celsius * 9.0 / 5.0 + 32.0:.2f}"


@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float) -> str:
    """Convert a Fahrenheit temperature to Celsius (C = (F - 32) * 5/9)."""
    return f"{(fahrenheit - 32.0) * 5.0 / 9.0:.2f}"


@mcp.tool()
def kilometers_to_miles(km: float) -> str:
    """Convert kilometers to miles (1 km = 0.621371 mile)."""
    return f"{km * 0.621371:.4f}"


@mcp.tool()
def miles_to_kilometers(miles: float) -> str:
    """Convert miles to kilometers (1 mile = 1.609344 km)."""
    return f"{miles * 1.609344:.4f}"


if __name__ == "__main__":
    mcp.run()  # stdio transport（默认）
