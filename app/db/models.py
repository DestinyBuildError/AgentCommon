"""SQLAlchemy 2.0 声明式模型 —— 严格按 的四张核心表。

- 所有主键为 TEXT（str(uuid4)）
- content / tool_input / result 使用 JSON 类型：SQLite 落文本序列化，
  切 PostgreSQL 时自动变 JSONB（with_variant，代码零改动）
- messages 表 UNIQUE(session_id, seq)：会话内单调递增，排序 + SSE 续传游标
- tool_calls 冗余审计表：熔断计数 / 成本分析直接扫表，带
  INDEX(session_id, tool_name, created_at)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# SQLite 用通用 JSON，PostgreSQL 用 JSONB（：SQLite → PostgreSQL 零改动切换）
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """统一 UTC 时间戳（naive）。

    SQLite 的 DateTime 落库丢 tzinfo、读回为 naive——模型层直接产出
    naive UTC，保证内存对象与 DB 往返值可比较、语义统一为 UTC。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Session(Base):
    """会话表（sessions）。status: idle / running / terminated。"""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # 预留多用户
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="idle", nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    effort: Mapped[str] = mapped_column(String(32), nullable=False)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 每轮累加
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    """对话记录 / 事件溯源主表（messages）。

    content 是完整 content blocks（含 tool_use / tool_result），不是纯文本——
    API 重放和 repair_orphans 的前提。UNIQUE(session_id, seq) 保证
    会话内 seq 单调唯一，作为 SSE 断线续传游标。
    """

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_messages_session_seq"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # 会话内单调递增
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, nullable=False)
    stop_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_read: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_creation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session: Mapped[Session] = relationship(back_populates="messages")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class ToolCall(Base):
    """工具调用审计（tool_calls）。冗余表是故意的。

    status: ok / error / timeout / denied / interrupted
    approval: auto / approved / denied / timeout
    """

    __tablename__ = "tool_calls"
    __table_args__ = (
        Index("ix_tool_calls_session_tool_created", "session_id", "tool_name", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), nullable=False)
    tool_use_id: Mapped[str] = mapped_column(String(64), nullable=False)  # 配对键
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_input: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    result: Mapped[Optional[Any]] = mapped_column(JSONType, nullable=True)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="ok", nullable=False)
    approval: Mapped[str] = mapped_column(String(16), default="auto", nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session: Mapped[Session] = relationship(back_populates="tool_calls")
    message: Mapped[Message] = relationship(back_populates="tool_calls")


class WorkspaceDir(Base):
    """工作区目录表（多目录工作区）。

    path 存规范化后的绝对路径（Path.resolve() 产物）；Windows 下大小写
    不敏感去重由 Repository 用 os.path.normcase 比较负责（UNIQUE 约束只
    兜底完全相同的字符串）。source: user（前端/API 添加或首次播种）/
    agent（模型经 add_workspace_dir 工具自主添加）。
    """

    __tablename__ = "workspace_dirs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class User(Base):
    """用户表（users，预留多用户扩展）。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class AppSetting(Base):
    """进程级键值设置（权限控制）：全局开关的持久化层。

    前端经 PUT /api/permissions 切换后即时写库——后端重启不丢用户选择；
    运行时权威在 SessionManager（内存），本表只负责跨重启恢复。
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
