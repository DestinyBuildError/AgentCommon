"""目录 demo server：password_gen（stdio 传输）—— 随机密码生成。

由 discover_tools 在工具目录中检索、install_mcp_server 以子进程 stdio 拉起
（见 tool_catalog.yaml）。提供 1 个工具：
  - generate_password(length=16, use_symbols=True, count=1)
    用 secrets 模块生成密码学安全的随机密码

协议与 demo_mcp_server.py 相同：MCP stdio（JSON-RPC over
stdin/stdout）。注意：stdio 传输独占 stdout——不要在这里 print() 调试信息。
"""

from __future__ import annotations

import secrets
import string

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="password_gen", instructions="Cryptographically secure password generator.")

_LOWER = string.ascii_lowercase
_UPPER = string.ascii_uppercase
_DIGITS = string.digits
_SYMBOLS = "!@#$%^&*-_=+?"


@mcp.tool()
def generate_password(length: int = 16, use_symbols: bool = True, count: int = 1) -> str:
    """Generate cryptographically secure random passwords.

    Args:
        length: Password length (8..128).
        use_symbols: Include punctuation symbols.
        count: How many passwords to generate (1..10).
    """
    length = max(8, min(128, int(length)))
    count = max(1, min(10, int(count)))
    alphabet = _LOWER + _UPPER + _DIGITS + (_SYMBOLS if use_symbols else "")
    passwords = ["".join(secrets.choice(alphabet) for _ in range(length)) for _ in range(count)]
    return "\n".join(passwords)


if __name__ == "__main__":
    mcp.run()  # stdio transport（默认）
