"""Repository —— 数据访问层，唯一允许触碰数据库的地方（分层职责）。

Agent 引擎与 API 层只依赖本类签名，不直接操作 SQLAlchemy 会话。
所有方法自包含事务：成功 commit，失败 rollback 后原样抛出异常。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AppSetting, Message, Session, ToolCall, WorkspaceDir, utcnow


class Repository:
    """异步数据访问门面。一个实例绑定一个 AsyncSession。"""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        model: str,
        effort: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Session:
        """创建会话，初始 status=idle，用量计数归零。"""
        session = Session(
            model=model,
            effort=effort,
            title=title,
            user_id=user_id,
            status="idle",
        )
        self._db.add(session)
        await self._db.commit()
        await self._db.refresh(session)
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        return await self._db.get(Session, session_id)

    async def list_sessions(
        self, limit: int = 100, offset: int = 0
    ) -> list[Session]:
        """会话列表（H5 抽屉）：最近活跃在前。"""
        result = await self._db.execute(
            select(Session)
            .order_by(Session.updated_at.desc(), Session.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部 messages / tool_calls（ORM 级联）。
        不存在返回 False。"""
        session = await self.get_session(session_id)
        if session is None:
            return False
        await self._db.delete(session)
        await self._db.commit()
        return True

    async def update_session_status(self, session_id: str, status: str) -> Optional[Session]:
        """更新会话状态：idle / running / terminated。不存在返回 None。"""
        session = await self.get_session(session_id)
        if session is None:
            return None
        session.status = status
        session.updated_at = utcnow()
        await self._db.commit()
        return session

    async def update_session_usage(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cost_usd: float = 0.0,
    ) -> Optional[Session]:
        """增量累加本轮用量与成本到 sessions 表（：每轮累加，列表页直接展示）。
        传入的是"本轮增量"，不是总量。不存在返回 None。"""
        session = await self.get_session(session_id)
        if session is None:
            return None
        session.total_input_tokens += input_tokens
        session.total_output_tokens += output_tokens
        session.cache_read_tokens += cache_read
        session.total_cost_usd += cost_usd
        session.updated_at = utcnow()
        await self._db.commit()
        return session

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        role: str,
        content_blocks: Sequence[Mapping[str, Any]],
        stop_reason: Optional[str] = None,
        usage: Optional[Mapping[str, Any]] = None,
    ) -> Message:
        """会话内追加一条消息，seq = 事务内 max(seq)+1，保证单调递增。

        usage 兼容两种键名：
        - Anthropic SDK 风格：input_tokens / output_tokens /
          cache_read_input_tokens / cache_creation_input_tokens
        - 列名风格：cache_read / cache_creation
        """
        seq_row = await self._db.execute(
            select(func.max(Message.seq)).where(Message.session_id == session_id)
        )
        current_max: int = seq_row.scalar_one() or 0

        def _usage_int(*keys: str) -> int:
            if not usage:
                return 0
            for key in keys:
                value = usage.get(key)
                if value is not None:
                    return int(value)
            return 0

        message = Message(
            session_id=session_id,
            seq=current_max + 1,
            role=role,
            content=[dict(block) for block in content_blocks],
            stop_reason=stop_reason,
            input_tokens=_usage_int("input_tokens"),
            output_tokens=_usage_int("output_tokens"),
            cache_read=_usage_int("cache_read_input_tokens", "cache_read"),
            cache_creation=_usage_int("cache_creation_input_tokens", "cache_creation"),
        )
        self._db.add(message)
        try:
            await self._db.commit()
        except IntegrityError:
            await self._db.rollback()
            raise
        await self._db.refresh(message)
        return message

    async def get_messages(self, session_id: str, after_seq: int = 0) -> list[Message]:
        """按 seq 升序取消息；after_seq>0 时只取增量（SSE/历史续传游标）。"""
        result = await self._db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.seq > after_seq)
            .order_by(Message.seq.asc())
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # tool_calls
    # ------------------------------------------------------------------

    async def record_tool_call(
        self,
        session_id: str,
        message_id: str,
        tool_use_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        result: Any = None,
        is_error: bool = False,
        status: str = "ok",
        approval: str = "auto",
        duration_ms: Optional[int] = None,
    ) -> ToolCall:
        """写入一条工具调用审计记录（：成功/失败/超时/拒绝/中断均落表）。"""
        call = ToolCall(
            session_id=session_id,
            message_id=message_id,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            tool_input=dict(tool_input),
            result=result,
            is_error=is_error,
            status=status,
            approval=approval,
            duration_ms=duration_ms,
        )
        self._db.add(call)
        await self._db.commit()
        await self._db.refresh(call)
        return call

    async def list_tool_calls(
        self, session_id: str, tool_name: Optional[str] = None
    ) -> list[ToolCall]:
        """审计查询：按时间升序；tool_name 过滤（熔断计数/成本分析扫表用）。"""
        stmt = select(ToolCall).where(ToolCall.session_id == session_id)
        if tool_name is not None:
            stmt = stmt.where(ToolCall.tool_name == tool_name)
        stmt = stmt.order_by(ToolCall.created_at.asc(), ToolCall.id.asc())
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # workspace_dirs（多目录工作区）
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(path: str) -> str:
        """Windows 大小写/斜杠不敏感比较键（os.path.normcase：nt 下小写化 +
        '/'→'\\'；posix 下恒等）。"""
        return os.path.normcase(path)

    async def list_workspace_dirs(self) -> list[WorkspaceDir]:
        """全部工作区目录，按加入时间升序（稳定列表 → SSE/API 顺序确定）。"""
        result = await self._db.execute(
            select(WorkspaceDir).order_by(WorkspaceDir.created_at.asc(), WorkspaceDir.id.asc())
        )
        return list(result.scalars().all())

    async def get_workspace_dir_by_path(self, path: str) -> Optional[WorkspaceDir]:
        """按路径查目录（normcase 比较：Windows 大小写不敏感）。"""
        norm = self._norm(path)
        for row in await self.list_workspace_dirs():
            if self._norm(row.path) == norm:
                return row
        return None

    async def add_workspace_dir(
        self, path: str, source: str = "user"
    ) -> tuple[WorkspaceDir, bool]:
        """新增工作区目录（path 须已是规范化绝对路径）。

        normcase 去重：已存在（Windows 大小写不敏感）时**不插入**，返回
        (既有行, False)；否则插入并返回 (新行, True)。并发插入撞 UNIQUE
        约束时回读既有行兜底。
        """
        existing = await self.get_workspace_dir_by_path(path)
        if existing is not None:
            return existing, False
        row = WorkspaceDir(path=path, source=source)
        self._db.add(row)
        try:
            await self._db.commit()
        except IntegrityError:
            await self._db.rollback()
            raced = await self.get_workspace_dir_by_path(path)
            if raced is not None:
                return raced, False
            raise
        await self._db.refresh(row)
        return row, True

    async def remove_workspace_dir(self, dir_id: str) -> bool:
        """按 id 删除工作区目录。不存在返回 False。

        「至少保留一个目录」的守卫在上层（WorkspaceRegistry）做——本层
        保持纯 CRUD 语义，便于复用与测试。
        """
        row = await self._db.get(WorkspaceDir, dir_id)
        if row is None:
            return False
        await self._db.delete(row)
        await self._db.commit()
        return True

    # ------------------------------------------------------------------
    # app_settings（全局开关持久化，如权限模式）
    # ------------------------------------------------------------------

    async def get_setting(self, key: str) -> str | None:
        """读一个键值设置；不存在返回 None。"""
        row = await self._db.get(AppSetting, key)
        return row.value if row is not None else None

    async def set_setting(self, key: str, value: str) -> None:
        """写入/覆盖一个键值设置（upsert）。"""
        row = await self._db.get(AppSetting, key)
        if row is None:
            row = AppSetting(key=key, value=value)
            self._db.add(row)
        else:
            row.value = value
        await self._db.commit()
