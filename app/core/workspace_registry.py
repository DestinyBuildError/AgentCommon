
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

WorkspaceChangeCallback = Callable[[str, WorkspaceDir, list[WorkspaceDir]], Awaitable[None]]


class WorkspaceError(Exception):

    status_code: int = 400

    def __init__(
        self, message: str, *, code: str = "workspace_error", hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class InvalidWorkspacePath(WorkspaceError):

    status_code = 400

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="invalid_path", hint=hint)


class DuplicateWorkspacePath(WorkspaceError):

    status_code = 409

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="duplicate_path", hint=hint)


class WorkspaceDirNotFound(WorkspaceError):

    status_code = 404

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="not_found", hint=hint)


class LastDirectoryProtected(WorkspaceError):

    status_code = 400

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="last_directory", hint=hint)


def workspace_dir_payload(record: WorkspaceDir) -> dict[str, Any]:
    return {
        "id": record.id,
        "path": record.path,
        "source": record.source,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


class WorkspaceRegistry:

    def __init__(
        self,
        session_factory: Any,
        *,
        on_change: WorkspaceChangeCallback | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._on_change = on_change
        self._records: list[WorkspaceDir] = []
        self._lock = asyncio.Lock()


    def get_dirs(self) -> list[Path]:
        return [Path(r.path) for r in self._records]

    def snapshot(self) -> list[WorkspaceDir]:
        return list(self._records)

    def is_empty(self) -> bool:
        return not self._records


    async def load(self) -> None:
        async with self._session_factory() as db:
            self._records = await Repository(db).list_workspace_dirs()
        logger.info("workspace registry loaded %d dir(s)", len(self._records))

    async def seed_if_empty(self, default_path: str | Path) -> bool:
        if not self.is_empty():
            return False
        resolved = str(Path(default_path).resolve())
        async with self._lock:
            record = await self._insert(resolved, source="user")
        logger.info("workspace seeded from config: %s", record.path)
        await self._notify("add", record)
        return True


    async def add(self, path: str, source: str = "user") -> WorkspaceDir:
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
            if not ok:
                raise WorkspaceDirNotFound(f"workspace directory not found: {dir_id}")
            self._records = [r for r in self._records if r.id != dir_id]
        logger.info("workspace dir removed: %s", record.path)
        await self._notify("remove", record)
        return record

    async def remove_by_ref(self, ref: str) -> WorkspaceDir:
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


    async def _insert(self, path: str, *, source: str) -> WorkspaceDir:
        async with self._session_factory() as db:
            record, created = await Repository(db).add_workspace_dir(path, source=source)
        if not created:
            raise DuplicateWorkspacePath(f"path already in workspace: {record.path}")
        self._records.append(record)
        return record

    async def _notify(self, action: str, record: WorkspaceDir) -> None:
        if self._on_change is None:
            return
        try:
            await self._on_change(action, record, self.snapshot())
        except Exception:
            logger.exception("workspace change broadcast failed (%s)", action)


def workspace_dir_payload_list(records: Sequence[WorkspaceDir]) -> list[dict[str, Any]]:
    return [workspace_dir_payload(r) for r in records]
