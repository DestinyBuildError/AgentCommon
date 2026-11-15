
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="unit_convert", instructions="Unit conversions: temperature & length.")


@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> str:
    return f"{celsius * 9.0 / 5.0 + 32.0:.2f}"


@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float) -> str:
    return f"{(fahrenheit - 32.0) * 5.0 / 9.0:.2f}"


@mcp.tool()
def kilometers_to_miles(km: float) -> str:
    return f"{km * 0.621371:.4f}"


@mcp.tool()
def miles_to_kilometers(miles: float) -> str:
    return f"{miles * 1.609344:.4f}"


if __name__ == "__main__":
    mcp.run()
