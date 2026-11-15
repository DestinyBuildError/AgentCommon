"""多目录工作区注册表：内存集合与 DB 双写同步 + 变更广播。

设计要点：
- **provider 模式**：read_file / search_workspace 注册时拿到的是
  ``roots_provider=registry.get_dirs``（闭包引用本服务），工具**执行时**
  实时查询当前目录集合——增删目录即刻生效，无需重注册/重启
- 内存集合（``_records``）是读路径的唯一来源；add/remove 先写 DB（经
  Repository，normcase 去重）成功后再动内存，失败即抛、两側永不漂移
- 变更后经 ``on_change`` 回调广播（main.py 接线 → 所有运行中会话的
  bus emit workspace_changed）；回调失败不影响变更本身（广播尽力而为，
  前端面板 refetch 兜底）
- 守卫：至少保留一个目录（last_directory）；新增目录必须是存在的绝对
  路径（invalid_path）；重复（Windows 大小写不敏感）拒绝（duplicate_path）

安全边界：本服务只管理"哪些目录对只读工具可见"，不提供任何写
能力；目录内的访问仍受 read_file/search_workspace 的 resolve + 包含判断
防穿越约束。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from app.db.models import WorkspaceDir
from app.db.repository import Repository

logger = logging.getLogger(__name__)

# on_change(action: "add"|"remove", record: WorkspaceDir, dirs: 当前完整列表)
WorkspaceChangeCallback = Callable[[str, WorkspaceDir, list[WorkspaceDir]], Awaitable[None]]


class WorkspaceError(Exception):
    """工作区管理失败基类。code 对外暴露（API 错误体 / 工具错误码）。"""

    status_code: int = 400

    def __init__(
        self, message: str, *, code: str = "workspace_error", hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class InvalidWorkspacePath(WorkspaceError):
    """不是绝对路径 / 不存在 / 不是目录 → API 400 invalid_path。"""

    status_code = 400

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="invalid_path", hint=hint)


class DuplicateWorkspacePath(WorkspaceError):
    """路径已在工作区（normcase 比较）→ API 409 duplicate_path。"""

    status_code = 409

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="duplicate_path", hint=hint)


class WorkspaceDirNotFound(WorkspaceError):
    """id / path 无法定位目录 → API 404 not_found。"""

    status_code = 404

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="not_found", hint=hint)


class LastDirectoryProtected(WorkspaceError):
    """最后一个目录不允许删除 → API 400 last_directory。"""

    status_code = 400

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="last_directory", hint=hint)


def workspace_dir_payload(record: WorkspaceDir) -> dict[str, Any]:
    """WorkspaceDir ORM 行 → 前后端契约的 dir 对象（tool 结果也复用）。"""
    return {
        "id": record.id,
        "path": record.path,
        "source": record.source,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


class WorkspaceRegistry:
    """进程内工作区目录服务：DB 持久 + 内存镜像 + 变更广播。"""

    def __init__(
        self,
        session_factory: Any,
        *,
        on_change: WorkspaceChangeCallback | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._on_change = on_change
        self._records: list[WorkspaceDir] = []
        self._lock = asyncio.Lock()  # add/remove 的读-改-写互斥

    # ------------------------------------------------------------------
    # 读路径（provider 模式：工具执行时实时查询，无重注册）
    # ------------------------------------------------------------------

    def get_dirs(self) -> list[Path]:
        """当前全部工作区目录（同步、无 await——工具 handler 直接调用）。"""
        return [Path(r.path) for r in self._records]

    def snapshot(self) -> list[WorkspaceDir]:
        """当前目录记录（API/SSE 载荷用）。"""
        return list(self._records)

    def is_empty(self) -> bool:
        return not self._records

    # ------------------------------------------------------------------
    # 启动：加载 + 播种
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """从 DB 全量加载到内存（lifespan 启动时调用一次）。"""
        async with self._session_factory() as db:
            self._records = await Repository(db).list_workspace_dirs()
        logger.info("workspace registry loaded %d dir(s)", len(self._records))

    async def seed_if_empty(self, default_path: str | Path) -> bool:
        """表为空时用 config workspace_root 播种（source=user）。

        播种**不做 isdir 校验**：config 里的种子目录允许尚不存在（与旧
        单目录行为兼容——旧实现同样不校验存在性，read_file 按需报 not_found）。
        返回是否真的播种了。
        """
        if not self.is_empty():
            return False
        resolved = str(Path(default_path).resolve())
        async with self._lock:
            record = await self._insert(resolved, source="user")
        logger.info("workspace seeded from config: %s", record.path)
        await self._notify("add", record)
        return True

    # ------------------------------------------------------------------
    # 变更（agent 工具与 API 共用入口）
    # ------------------------------------------------------------------

    async def add(self, path: str, source: str = "user") -> WorkspaceDir:
        """新增目录：校验（绝对路径 + 存在且为目录 + normcase 去重）→
        落库 → 更新内存 → 广播。失败抛 WorkspaceError 子类。"""
        raw = (path or "").strip()
        if not raw:
            raise InvalidWorkspacePath(
                "path is required",
                hint="Provide an absolute directory path, e.g. D:\\data",
            )
        if not os.path.isabs(raw):
            raise InvalidWorkspacePath(
                f"not an absolute path: {raw}",
                hint="Provide an absolute directory path, e.g. D:\\data",
            )
        resolved = Path(raw).resolve()
        if not resolved.is_dir():
            raise InvalidWorkspacePath(
                f"path does not exist or is not a directory: {raw}",
                hint="Only existing directories can be added to the workspace",
            )
        norm = os.path.normcase(str(resolved))
        async with self._lock:
            for record in self._records:
                if os.path.normcase(record.path) == norm:
                    raise DuplicateWorkspacePath(
                        f"path already in workspace: {record.path}",
                        hint="Each directory can only be added once",
                    )
            record = await self._insert(str(resolved), source=source)
        logger.info("workspace dir added (%s): %s", source, record.path)
        await self._notify("add", record)
        return record

    async def remove(self, dir_id: str) -> WorkspaceDir:
        """按 id 删除目录。守卫：至少保留一个（last_directory）。
        落库 → 更新内存 → 广播。"""
        async with self._lock:
            record = next((r for r in self._records if r.id == dir_id), None)
            if record is None:
                raise WorkspaceDirNotFound(
                    f"workspace directory not found: {dir_id}",
                    hint="GET /api/workspaces or list_workspace_dirs for current ids",
                )
            if len(self._records) <= 1:
                raise LastDirectoryProtected(
                    "cannot remove the last workspace directory",
                    hint="Add another directory before removing this one",
                )
            async with self._session_factory() as db:
                ok = await Repository(db).remove_workspace_dir(dir_id)
            if not ok:  # pragma: no cover - 内存与 DB 失步的防御
                raise WorkspaceDirNotFound(f"workspace directory not found: {dir_id}")
            self._records = [r for r in self._records if r.id != dir_id]
        logger.info("workspace dir removed: %s", record.path)
        await self._notify("remove", record)
        return record

    async def remove_by_ref(self, ref: str) -> WorkspaceDir:
        """按 id 或路径（normcase 不敏感）删除目录——remove_workspace_dir
        工具的入口；API 端点用纯 id 的 remove()。"""
        text = (ref or "").strip()
        record = next((r for r in self.snapshot() if r.id == text), None)
        if record is None:
            norm = os.path.normcase(text)
            record = next(
                (r for r in self.snapshot() if os.path.normcase(r.path) == norm), None
            )
        if record is None:
            raise WorkspaceDirNotFound(
                f"workspace directory not found: {text}",
                hint="Pass a directory id or path from list_workspace_dirs",
            )
        return await self.remove(record.id)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _insert(self, path: str, *, source: str) -> WorkspaceDir:
        """落库并追加进内存（调用方须已持锁或处于启动单线程上下文）。"""
        async with self._session_factory() as db:
            record, created = await Repository(db).add_workspace_dir(path, source=source)
        if not created:  # 并发路径下 DB 层去重兜底
            raise DuplicateWorkspacePath(f"path already in workspace: {record.path}")
        self._records.append(record)
        return record

    async def _notify(self, action: str, record: WorkspaceDir) -> None:
        """变更广播（尽力而为：回调失败只记日志，不影响变更结果）。"""
        if self._on_change is None:
            return
        try:
            await self._on_change(action, record, self.snapshot())
        except Exception:  # pragma: no cover - 防御：广播不阻断管理操作
            logger.exception("workspace change broadcast failed (%s)", action)


def workspace_dir_payload_list(records: Sequence[WorkspaceDir]) -> list[dict[str, Any]]:
    """记录列表 → 契约的 dirs 数组。"""
    return [workspace_dir_payload(r) for r in records]
