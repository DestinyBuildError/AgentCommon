"""Skills 渐进加载。

机制：
- 启动时扫描 skills/*/SKILL.md，解析 YAML frontmatter（name/description）+
  正文；把**一行描述**注入 system prompt（按 name 排序 → 稳定内容 → 可缓存，
  缓存纪律：system prompt 冻结、不插时间戳）
- 模型经 load_skill(name) 工具按需加载全文；SkillNotFound → is_error +
  可用 skill 列表（模型自纠）
- **上下文陷阱**：skill 正文几千 token，加载后永久占对话历史——同 skill
  重复加载直接返回缓存副本引用（省 token，正文以首次加载的 tool_result
  为准，仍在对话历史里）

frontmatter 约定（SKILL.md）：
    ---
    name: code_review # 必填，与目录名一致时最稳
    description: 一行描述 # 必填，进 system prompt
    ---
    正文（markdown，数千字内）

frontmatter 缺失/不合法/缺 name 的文件跳过 + warning，不阻塞其余 skill。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.tools.errors import ToolPermanentError, ToolInputError
from app.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

# 重复加载时的缓存副本引用文本（正文已在对话历史，不再重复注入）
_CACHE_NOTE_TEMPLATE = (
    "[skill '{name}' already loaded earlier in this conversation — "
    "refer to the full text in the earlier tool_result above instead of "
    "loading it again. This cached reference is returned to save tokens.]"
)


@dataclass(frozen=True)
class Skill:
    """单个 skill：frontmatter 元数据 + 正文。"""

    name: str
    description: str
    body: str          # frontmatter 之后的 markdown 正文
    path: Path

    @property
    def full_text(self) -> str:
        """加载进对话的全文（frontmatter + 正文，原样保序）。"""
        return (
            f"---\nname: {self.name}\ndescription: {self.description}\n---\n{self.body}"
        )


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """解析 frontmatter（首个 --- ... --- 块）+ 正文。无 frontmatter 返回 ({}, 原文)。"""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            raw = "".join(lines[1:idx])
            body = "".join(lines[idx + 1:])
            try:
                meta = yaml.safe_load(raw)
            except yaml.YAMLError:
                raise ValueError("invalid YAML frontmatter") from None
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                raise ValueError("frontmatter must be a mapping")
            return meta, body
    raise ValueError("unterminated frontmatter (missing closing '---')")


def load_skill_file(path: Path) -> Skill | None:
    """读取单个 SKILL.md → Skill；不合法返回 None（caller 记 warning）。"""
    text = path.read_text(encoding="utf-8")
    try:
        meta, body = parse_skill_md(text)
    except ValueError as exc:
        logger.warning("skill file %s skipped: %s", path, exc)
        return None
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        logger.warning("skill file %s skipped: frontmatter missing 'name'", path)
        return None
    return Skill(name=name, description=description, body=body, path=path)


def load_skills_from_dir(root: Path) -> dict[str, Skill]:
    """扫描 skills/*/SKILL.md → {name: Skill}（name 冲突后者覆盖 + warning）。

    目录不存在 / 为空 → 返回 {}（skills 是可选能力，不阻塞启动）。
    """
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills
    for path in sorted(root.glob(f"*/{SKILL_FILENAME}")):
        skill = load_skill_file(path)
        if skill is None:
            continue
        if skill.name in skills:
            logger.warning("duplicate skill name %r (%s); overriding", skill.name, path)
        skills[skill.name] = skill
    logger.info("loaded %d skill(s) from %s", len(skills), root)
    return skills


def build_prompt_section(skills: dict[str, Skill]) -> str:
    """system prompt 的 skills 段。

    按 name 排序 → 内容稳定 → prompt caching 前缀不变。
    无 skill 返回空串（system prompt 保持原样）。
    """
    if not skills:
        return ""
    lines = [
        "",
        "",
        "## Skills",
        "以下 skill 可用 load_skill(name) 工具按需加载全文（仅一行描述常驻，控制上下文）：",
    ]
    for name in sorted(skills):
        lines.append(f"- {name}: {skills[name].description}")
    return "\n".join(lines)


class SkillLoader:
    """load_skill 工具的执行体：全文返回 + 已加载缓存（上下文陷阱）。"""

    def __init__(self, skills: dict[str, Skill]) -> None:
        self._skills = dict(skills)
        self._loaded: set[str] = set()

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def load(self, name: str) -> str:
        skill = self._skills.get(name.strip())
        if skill is None:
            raise ToolPermanentError(
                f"Skill not found: {name.strip()}",
                code="skill_not_found",
                hint=f"Available skills: {self.names or '(none)'}",
            )
        if name in self._loaded:
            return _CACHE_NOTE_TEMPLATE.format(name=name)
        self._loaded.add(name)
        return skill.full_text

    def reset_cache(self) -> None:
        """清空已加载缓存（测试 / 会话轮换用；进程级缓存保留也无碍正确性）。"""
        self._loaded.clear()


def register_load_skill_tool(
    registry: ToolRegistry, skills: dict[str, Skill]
) -> SkillLoader:
    """注册 load_skill 工具。返回 loader（测试断言缓存用）。

    无 skill 时也注册（模型调用得到明确的 skill_not_found，优于幻觉）。
    """
    loader = SkillLoader(skills)

    async def handler(tool_input: dict[str, Any]) -> str:
        name = tool_input.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolInputError(
                "load_skill requires non-empty string field 'name'",
                hint=f"Available skills: {loader.names or '(none)'}",
            )
        return loader.load(name)

    registry.register(
        ToolSpec(
            name="load_skill",
            description=(
                "Load the full text of a named skill (playbook). "
                "Descriptions of available skills are listed in the system prompt."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name, e.g. 'code_review'"},
                },
                "required": ["name"],
            },
            handler=handler,
            timeout_s=5.0,
            parallel_safe=True,
            dangerous=False,
        )
    )
    return loader
