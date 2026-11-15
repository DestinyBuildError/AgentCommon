
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.mcp_loader import MCPServerConfig

logger = logging.getLogger(__name__)

TRUST_LEVELS = ("builtin", "community")
DEFAULT_TRUST = "community"
SEARCH_TOP_K = 5


@dataclass(frozen=True)
class CatalogItem:

    name: str
    description: str
    keywords: tuple[str, ...] = ()
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    trust: str = DEFAULT_TRUST
    cwd: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CatalogItem:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("catalog entry missing required field 'name'")
        description = str(raw.get("description") or "").strip()
        if not description:
            raise ValueError(f"catalog[{name}]: missing required field 'description'")
        transport = raw.get("transport")
        if not isinstance(transport, dict):
            raise ValueError(f"catalog[{name}]: missing 'transport' mapping")
        command = transport.get("command")
        url = transport.get("url")
        if not command and not url:
            raise ValueError(f"catalog[{name}]: transport needs 'command' (stdio) or 'url' (http)")
        args_raw = transport.get("args") or []
        if not isinstance(args_raw, list):
            raise ValueError(f"catalog[{name}]: transport.args must be a list")
        keywords_raw = raw.get("keywords") or []
        if not isinstance(keywords_raw, list):
            raise ValueError(f"catalog[{name}]: 'keywords' must be a list")
        trust = str(raw.get("trust") or DEFAULT_TRUST).strip().lower()
        if trust not in TRUST_LEVELS:
            raise ValueError(f"catalog[{name}]: trust must be one of {TRUST_LEVELS}")
        return cls(
            name=name,
            description=description,
            keywords=tuple(str(k) for k in keywords_raw),
            command=str(command) if command else None,
            args=tuple(str(a) for a in args_raw),
            url=str(url) if url else None,
            trust=trust,
        )

    def to_server_config(self) -> MCPServerConfig:
        return MCPServerConfig(
            name=self.name,
            command=self.command,
            args=list(self.args),
            url=self.url,
            cwd=self.cwd,
            enabled=True,
        )

    def match_score(self, query: str) -> int:
        q = (query or "").strip().lower()
        if not q:
            return 0
        name_l = self.name.lower()
        desc_l = self.description.lower()
        kws_l = [k.lower() for k in self.keywords if k]
        probes: list[str] = []
        candidates = [q, *(t for t in q.split() if self._meaningful(t))]
        for probe in candidates:
            if probe and probe not in probes:
                probes.append(probe)
        score = 0
        for probe in probes:
            if probe in name_l:
                score += 4
            score += 2 * sum(
                1 for k in kws_l if probe in k or (len(k) >= 2 and k in probe)
            )
            if probe in desc_l:
                score += 1
        return score

    @staticmethod
    def _meaningful(token: str) -> bool:
        if len(token) >= 3:
            return True
        return any("一" <= ch <= "鿿" for ch in token)


class ToolCatalog:

    def __init__(self, items: list[CatalogItem]) -> None:
        self._items: dict[str, CatalogItem] = {}
        for item in items:
            if item.name in self._items:
                logger.warning("duplicate catalog entry '%s'; first one wins", item.name)
                continue
            self._items[item.name] = item

    def get(self, name: str) -> CatalogItem | None:
        return self._items.get(name)

    def names(self) -> list[str]:
        return sorted(self._items)

    def all(self) -> list[CatalogItem]:
        return [self._items[name] for name in sorted(self._items)]

    def search(
        self,
        query: str,
        installed: set[str] | None = None,
        *,
        top_k: int = SEARCH_TOP_K,
    ) -> list[dict[str, Any]]:
        installed = installed or set()
        scored = [
            (item.match_score(query), item)
            for item in self._items.values()
        ]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [
            {
                "name": item.name,
                "description": item.description,
                "trust": item.trust,
                "installed": item.name in installed,
            }
            for _, item in scored[:top_k]
        ]


def load_tool_catalog(path: str | Path) -> ToolCatalog:
    p = Path(path)
    if not p.exists():
        logger.warning("tool catalog not found at %s; discover_tools will find nothing", p)
        return ToolCatalog([])
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to parse tool catalog %s; ignored: %r", p, exc)
        return ToolCatalog([])
    if not isinstance(data, list):
        logger.warning("tool catalog %s must be a list; ignoring", p)
        return ToolCatalog([])
    base_dir = str(p.resolve().parent)
    items: list[CatalogItem] = []
    for raw in data:
        try:
            if not isinstance(raw, dict):
                raise ValueError("catalog entry must be a mapping")
            item = CatalogItem.from_dict(raw)
        except ValueError as exc:
            logger.warning("invalid catalog entry skipped: %s", exc)
            continue
        items.append(replace(item, cwd=base_dir))
    return ToolCatalog(items)
