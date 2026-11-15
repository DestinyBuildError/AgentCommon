
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

_CACHE_NOTE_TEMPLATE = (
    "[skill '{name}' already loaded earlier in this conversation — "
    "refer to the full text in the earlier tool_result above instead of "
    "loading it again. This cached reference is returned to save tokens.]"
)


@dataclass(frozen=True)
class Skill:

    name: str
    description: str
    body: str
    path: Path

    @property
    def full_text(self) -> str:
        return (
            f"---\nname: {self.name}\ndescription: {self.description}\n---\n{self.body}"
        )


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
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
        self._loaded.clear()


def register_load_skill_tool(
    registry: ToolRegistry, skills: dict[str, Skill]
) -> SkillLoader:
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
