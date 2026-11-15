"""精选工具目录（自扩展 Agent）——discover_tools 的搜索空间。

tool_catalog.yaml 每项字段：
    name server 名（安装后作为 mcp__<name>__ 前缀的命名空间）
    description 一句话描述（匹配目标 + 安装后成为模型可见的工具集说明线索）
    keywords 检索关键词（可中英混排）
    transport {command, args} → stdio 子进程 或 {url} → streamable http
    trust builtin（随仓库分发、已审计）| community（第三方，安装需审批）
                 缺省按 community 处理（保守信任模型）

解析纪律与 parse_mcp_servers 一致：非法项跳过 + warning，绝不阻塞启动；
目录文件缺失 → 空目录（discover_tools 返回无结果，其余功能不受影响）。
transport.command 的裸 python 别名由 MCPManager 的别名解析统一处理；
相对 args 路径在装载时锚定到**目录文件所在目录**，与进程 CWD 无关。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.mcp_loader import MCPServerConfig

logger = logging.getLogger(__name__)

TRUST_LEVELS = ("builtin", "community")
DEFAULT_TRUST = "community"
SEARCH_TOP_K = 5  # discover_tools 默认返回条数


@dataclass(frozen=True)
class CatalogItem:
    """单个可安装 MCP server 的目录描述。"""

    name: str
    description: str
    keywords: tuple[str, ...] = ()
    command: str | None = None       # stdio 模式：可执行文件
    args: tuple[str, ...] = ()
    url: str | None = None           # http 模式：streamable http 端点
    trust: str = DEFAULT_TRUST
    cwd: str | None = None           # stdio 子进程工作目录（装载时锚定目录基址）

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
        """目录项 → 运行时安装用的 MCP 连接配置（enabled 恒真）。"""
        return MCPServerConfig(
            name=self.name,
            command=self.command,
            args=list(self.args),
            url=self.url,
            cwd=self.cwd,
            enabled=True,
        )

    def match_score(self, query: str) -> int:
        """查询 → 相关性得分（0 = 不相关）。

        大小写不敏感；整条查询与逐 token 双重探测。关键词命中是**双向**
        包含：探测串出现在关键词里（"reverse" 命中 "reversed"）或关键词
        出现在探测串里（中文无空格短语 "温度换算" 命中关键词 "温度"，
        词干形态 "converting" 命中 "convert"；反向要求关键词 ≥2 字符防
        单字误命中）。name 命中权重最高，keywords 次之，description 最低
        ——同分按名字排序保证结果确定性。
        """
        q = (query or "").strip().lower()
        if not q:
            return 0
        name_l = self.name.lower()
        desc_l = self.description.lower()
        kws_l = [k.lower() for k in self.keywords if k]
        # 整条查询恒参与；分词过滤英文短 token（"to"/"of" 之类停用词会
        # 误命中 "generator" 的 "to"），中文两字词（"密码"/"温度"）保留
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
        """token 是否值得参与匹配：≥3 字符，或含 CJK（中文两字词有效）。"""
        if len(token) >= 3:
            return True
        return any("一" <= ch <= "鿿" for ch in token)


class ToolCatalog:
    """目录容器：按名取用 + 关键词搜索。重名条目先到先得 + warning。"""

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
        """全部条目（按名排序，确定性）。"""
        return [self._items[name] for name in sorted(self._items)]

    def search(
        self,
        query: str,
        installed: set[str] | None = None,
        *,
        top_k: int = SEARCH_TOP_K,
    ) -> list[dict[str, Any]]:
        """关键词搜索：得分 > 0 的条目按 (-score, name) 排序取前 top_k。

        返回形态（discover_tools 的 payload 元素）：
            {name, description, trust, installed}
        """
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
    """装载目录文件；缺失/损坏/非法项一律降级（warning + 跳过），不阻塞启动。

    stdio transport 的相对路径锚定到目录文件所在目录——
    记入 CatalogItem.cwd，与后端进程的实际 CWD 解耦。
    """
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
