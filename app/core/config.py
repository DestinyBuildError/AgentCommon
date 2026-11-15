"""全局配置：pydantic-settings，环境变量驱动。

字段与 config.yaml.example 一一对应。
密钥（anthropic_api_key）只应来自环境变量，禁止写入 config.yaml 提交仓库。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """后端全局配置。环境变量优先；可选 .env 文件兜底。"""

    # ---- LLM ----
    anthropic_api_key: str = ""
    model: str = "claude-opus-4-8"
    effort: str = "high"          # low / medium / high
    max_tokens: int = 16000

    # ---- Agent 引擎安全阀 ----
    tool_breaker_limit: int = 3      # 同一工具连续失败 >= N 次 → 熔断
    max_iterations: int = 30         # 单轮 agent loop 最大迭代
    approval_timeout_s: int = 120    # 审批超时自动 deny
    session_idle_timeout_s: int = 600  # 会话空闲回收（10 分钟）

    # ---- 任务拆解规划器（后端思考增强）----
    # medium/complex 任务先发 task_options 让用户选执行方向；超时自动采用
    # 推荐方案。False → 消息直接进 agent loop，规划器完全不参与
    planner_enabled: bool = True
    planner_max_options: int = 4     # 选项数上限（2~4）
    planner_timeout_s: int = 120     # 用户选择超时（秒）→ 自动采用推荐方案
    # 长时任务进度推送：规划器标记或运行超阈值自动升级后，
    # 每 progress_interval_s 发一次 task_progress SSE
    progress_interval_s: int = 60    # 进度推送间隔（秒）
    long_running_after_s: int = 90   # 运行超过该秒数自动升级为长时任务

    # ---- 存储 ----
    database_url: str = "sqlite+aiosqlite:///./data/agent.db"
    # 首次播种种子：workspace_dirs 表为空时作为第一个工作区目录（source=user）；
    # 运行期目录由 /api/workspaces 或 agent 工具动态增删（多目录工作区）
    workspace_root: str = "../workspace"
    agent_manage_dirs: bool = True   # False → 禁止 agent 自主增删工作区目录（工具返回 is_error）

    # ---- MCP（）----
    mcp_enabled: bool = True          # False → 完全跳过 MCP 接入（测试/降级开关）
    mcp_servers: list[dict[str, Any]] = []  # 见 config.yaml.example 的 mcp_servers 段

    # ---- 自扩展 Agent（发现/安装/卸载 MCP server + Python 沙箱）----
    # False → 四个元工具（discover_tools / install_mcp_server /
    # uninstall_mcp_server / run_python）执行时统一返回 is_error
    enable_tool_discovery: bool = True
    # 精选工具目录（discover_tools 的搜索空间）；相对路径相对后端进程 CWD
    tool_catalog_path: str = "tool_catalog.yaml"
    # run_python 单次执行超时上限（秒）——模型传更大的 timeout_s 会被钳制到该值
    run_python_max_timeout_s: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    # 测试隔离开关：env LOAD_CONFIG_YAML=false 时 get_settings() 跳过 yaml，
    # 测试套件不依赖用户编辑的 config.yaml 内容（conftest 顶部 setdefault）
    load_config_yaml: bool = True


@lru_cache
def get_settings() -> Settings:
    """进程级单例配置。

    优先级：环境变量 > config.yaml > .env > 字段默认值。
    yaml 覆盖项在构造时过滤掉"环境变量已显式设置"的键，
    保证 env 永远压过 yaml（与模块 docstring 的约定一致）。
    """
    import os

    base = Settings()
    if not base.load_config_yaml:
        return base
    overrides = load_yaml_overrides(Path("config.yaml"))
    env_wins = {
        k: v
        for k, v in overrides.items()
        if os.environ.get(k.upper()) is None
    }
    return Settings(**env_wins)


def load_yaml_overrides(path: str | Path) -> dict[str, Any]:
    """读取 config.yaml 中与 Settings 字段同名的键，返回合法覆盖项。

    供后续波次把 config.yaml（模型、工具开关、MCP 服务器列表）接入
    Settings 使用；文件不存在或为空时返回 {}。anthropic_api_key 即使
    出现在 yaml 中也会被过滤——密钥只走环境变量。
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    allowed = set(Settings.model_fields) - {"anthropic_api_key"}
    return {k: v for k, v in data.items() if k in allowed and v is not None}
