
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
    length = max(8, min(128, int(length)))
    count = max(1, min(10, int(count)))
    alphabet = _LOWER + _UPPER + _DIGITS + (_SYMBOLS if use_symbols else "")
    passwords = ["".join(secrets.choice(alphabet) for _ in range(length)) for _ in range(count)]
    return "\n".join(passwords)


if __name__ == "__main__":
    mcp.run()
