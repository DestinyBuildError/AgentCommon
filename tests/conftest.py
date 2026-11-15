"""测试公共配置：确保从任意工作目录运行 pytest 都能导入 app 包。"""

import os
import sys
from pathlib import Path

# API 测试会真实跑 lifespan——禁掉 MCP 接入，避免开发者本地 config.yaml
# 里配置的 MCP server 在每次测试中 spawn 子进程（MCP 自身的行为由
# test_mcp_loader.py 以注入式 fake / 显式标记的 demo 端到端用例覆盖）。
# 必须在 get_settings() 首次调用（lru_cache）前设置。
os.environ.setdefault("MCP_ENABLED", "false")
# 同理：测试不依赖用户编辑的 config.yaml 内容（model/workspace_root 等随时会被
# 用户改动）——跳过 yaml 装载，用 Settings 字段默认值保证测试封闭性。
os.environ.setdefault("LOAD_CONFIG_YAML", "false")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
