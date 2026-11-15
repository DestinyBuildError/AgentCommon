
import os
import sys
from pathlib import Path

os.environ.setdefault("MCP_ENABLED", "false")
os.environ.setdefault("LOAD_CONFIG_YAML", "false")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
