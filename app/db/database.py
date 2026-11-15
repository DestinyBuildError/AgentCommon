"""异步引擎与会话工厂。

- DATABASE_URL 来自 config，默认 sqlite+aiosqlite:///./data/agent.db
- SQLite 文件路径的父目录自动创建（data/）
- get_db：FastAPI 请求级依赖
- init_db：开发期 create_all 建表；生产换 Alembic 迁移
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """sqlite:///./data/agent.db → 确保 data/ 存在；内存库跳过。"""
    if not database_url.startswith("sqlite"):
        return
    if ":memory:" in database_url or "mode=memory" in database_url:
        return
    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        prefix = "sqlite:///"
    file_path = database_url[len(prefix):].split("?", 1)[0]
    if not file_path or file_path == ":memory:":
        return
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def build_engine(database_url: str | None = None) -> AsyncEngine:
    """按 URL 构建异步引擎。测试可直接传入内存库 URL 复用本函数。"""
    url = database_url if database_url is not None else get_settings().database_url
    _ensure_sqlite_parent_dir(url)
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            # 内存库必须共享单连接，否则每个连接各拿到一个空库
            kwargs["poolclass"] = StaticPool
    return create_async_engine(url, **kwargs)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, autoflush=False
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：请求级 AsyncSession（用完即关）。"""
    async with get_sessionmaker()() as session:
        yield session


async def init_db() -> None:
    """建表（开发期 create_all）。启动时由 main.lifespan 调用；
    孤儿 tool_use 修复（repair_orphans）由后续波次在启动流程中跟进。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """释放连接池（应用关闭 / 测试收尾）。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
