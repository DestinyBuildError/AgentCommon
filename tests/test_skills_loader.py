"""skills_loader 测试：扫描 / frontmatter 解析 / 全文加载 /
SkillNotFound（is_error + 可用列表）/ 重复加载缓存复用 / prompt 段稳定。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.dispatcher import ToolDispatcher
from app.skills_loader import (
    Skill,
    build_prompt_section,
    load_skill_file,
    load_skills_from_dir,
    parse_skill_md,
    register_load_skill_tool,
)
from app.tools.registry import ToolRegistry


def write_skill(root: Path, name: str, fm_name: str, description: str, body: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {fm_name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    write_skill(
        root, "code_review", "code_review", "系统化代码审查流程", "# 审查流程\n\n第一步……"
    )
    write_skill(
        root, "data_analysis", "data_analysis", "数据分析四步法", "# 数据分析\n\n体检……"
    )
    return root


# ----------------------------------------------------------------------
# frontmatter 解析
# ----------------------------------------------------------------------


def test_parse_skill_md_splits_frontmatter_and_body() -> None:
    meta, body = parse_skill_md("---\nname: a\ndescription: b\n---\n正文第一行\n正文第二行")
    assert meta == {"name": "a", "description": "b"}
    assert body == "正文第一行\n正文第二行"


def test_parse_skill_md_no_frontmatter_returns_raw() -> None:
    meta, body = parse_skill_md("没有 frontmatter 的纯文本")
    assert meta == {}
    assert body == "没有 frontmatter 的纯文本"


def test_parse_skill_md_invalid_frontmatter_raises() -> None:
    with pytest.raises(ValueError):
        parse_skill_md("---\nname: [未闭合\n---\nbody")  # yaml 语法错
    with pytest.raises(ValueError):
        parse_skill_md("---\nname: a\n")  # 未闭合（无收尾 ---）


def test_load_skill_file_missing_name_skipped(tmp_path: Path) -> None:
    p = tmp_path / "SKILL.md"
    p.write_text("---\ndescription: no name here\n---\nbody", encoding="utf-8")
    assert load_skill_file(p) is None


# ----------------------------------------------------------------------
# 扫描
# ----------------------------------------------------------------------


def test_load_skills_from_dir_scans_and_maps(skills_dir: Path) -> None:
    skills = load_skills_from_dir(skills_dir)
    assert set(skills) == {"code_review", "data_analysis"}
    assert skills["code_review"].description == "系统化代码审查流程"
    assert "审查流程" in skills["code_review"].body
    # full_text 保序含 frontmatter + 正文
    full = skills["code_review"].full_text
    assert full.startswith("---\nname: code_review")
    assert "# 审查流程" in full


def test_load_skills_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_skills_from_dir(tmp_path / "nope") == {}


def test_load_skills_invalid_file_skipped_others_kept(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "good", "good", "ok", "正文")
    bad = root / "bad" / "SKILL.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("没有 frontmatter", encoding="utf-8")  # 缺 frontmatter → 跳过
    skills = load_skills_from_dir(root)
    assert set(skills) == {"good"}


# ----------------------------------------------------------------------
# system prompt 注入段（稳定排序 → 可缓存）
# ----------------------------------------------------------------------


def test_prompt_section_sorted_and_stable(skills_dir: Path) -> None:
    skills = load_skills_from_dir(skills_dir)
    s1 = build_prompt_section(skills)
    s2 = build_prompt_section(skills)
    assert s1 == s2  # 两次构建逐字节一致（缓存纪律）
    lines = [ln for ln in s1.splitlines() if ln.startswith("- ")]
    assert lines == [
        "- code_review: 系统化代码审查流程",
        "- data_analysis: 数据分析四步法",
    ]  # 按 name 排序，一行描述常驻


def test_prompt_section_empty_skills_returns_empty_string() -> None:
    assert build_prompt_section({}) == ""


# ----------------------------------------------------------------------
# load_skill 工具：加载 / 未找到 / 缓存复用
# ----------------------------------------------------------------------


def make_dispatcher(skills_dir: Path) -> tuple[ToolDispatcher, object]:
    registry = ToolRegistry()
    loader = register_load_skill_tool(registry, load_skills_from_dir(skills_dir))
    return ToolDispatcher(registry), loader


async def test_load_skill_tool_full_text_then_cache(skills_dir: Path) -> None:
    dispatcher, loader = make_dispatcher(skills_dir)

    first = await dispatcher.dispatch("load_skill", {"name": "code_review"}, "tu_1")
    assert first.is_error is False
    assert first.content.startswith("---\nname: code_review")
    assert "# 审查流程" in first.content

    # 同 run 重复加载 → 缓存副本引用（不含正文，省 token）
    second = await dispatcher.dispatch("load_skill", {"name": "code_review"}, "tu_2")
    assert second.is_error is False
    assert "already loaded" in second.content
    assert "审查流程" not in second.content

    # 另一个 skill 首次加载仍是全文；缓存按 skill 独立
    third = await dispatcher.dispatch("load_skill", {"name": "data_analysis"}, "tu_3")
    assert "数据分析" in third.content
    assert loader.reset_cache() is None
    fourth = await dispatcher.dispatch("load_skill", {"name": "code_review"}, "tu_4")
    assert fourth.content == first.content  # 清缓存后重新全文加载


async def test_cache_note_far_cheaper_than_full_text() -> None:
    """真实体量的 skill（几千字正文）重复加载只返回一行引用——省 token。"""
    registry = ToolRegistry()
    big_body = "# 大型 skill\n\n" + ("分析步骤详述。\n\n" * 400)  # ~6k 字符
    skill = Skill(name="big", description="大 skill", body=big_body, path=Path("."))
    loader = register_load_skill_tool(registry, {"big": skill})
    dispatcher = ToolDispatcher(registry)

    first = await dispatcher.dispatch("load_skill", {"name": "big"}, "tu_1")
    second = await dispatcher.dispatch("load_skill", {"name": "big"}, "tu_2")
    assert not first.is_error and not second.is_error
    assert len(first.content) > 3000
    assert len(second.content) < 300  # 引用远小于全文
    assert loader.names == ["big"]


async def test_load_skill_not_found_is_error_with_available_list(skills_dir: Path) -> None:
    dispatcher, _ = make_dispatcher(skills_dir)
    result = await dispatcher.dispatch("load_skill", {"name": "ghost"}, "tu_5")
    assert result.is_error is True
    assert result.code == "skill_not_found"
    assert "Available skills" in result.content
    assert "code_review" in result.content  # hint 里列出可用 skill


async def test_load_skill_missing_param_is_error(skills_dir: Path) -> None:
    dispatcher, _ = make_dispatcher(skills_dir)
    result = await dispatcher.dispatch("load_skill", {}, "tu_6")
    assert result.is_error is True
    assert result.code == "invalid_params"


def test_skill_dataclass_full_text_format() -> None:
    skill = Skill(name="x", description="d", body="正文", path=Path("."))
    assert skill.full_text == "---\nname: x\ndescription: d\n---\n正文"
