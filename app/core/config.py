
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    anthropic_api_key: str = ""
    model: str = "claude-opus-4-8"
    effort: str = "high"
    max_tokens: int = 16000

    tool_breaker_limit: int = 3
    max_iterations: int = 30
    approval_timeout_s: int = 120
    session_idle_timeout_s: int = 600

    planner_enabled: bool = True
    planner_max_options: int = 4
    planner_timeout_s: int = 120
    progress_interval_s: int = 60
    long_running_after_s: int = 90

    database_url: str = "sqlite+aiosqlite:///./data/agent.db"
    workspace_root: str = "../workspace"
    agent_manage_dirs: bool = True

    mcp_enabled: bool = True
    mcp_servers: list[dict[str, Any]] = []

    enable_tool_discovery: bool = True
    tool_catalog_path: str = "tool_catalog.yaml"
    run_python_max_timeout_s: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    load_config_yaml: bool = True


@lru_cache
def get_settings() -> Settings:
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
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    allowed = set(Settings.model_fields) - {"anthropic_api_key"}
    return {k: v for k, v in data.items() if k in allowed and v is not None}
