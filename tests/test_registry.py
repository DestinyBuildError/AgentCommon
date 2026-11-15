
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.tools.builtin import register_default_tools
from app.tools.errors import ToolInputError, ToolPermanentError
from app.tools.registry import ToolRegistry, ToolSpec


def make_spec(name: str, **overrides) -> ToolSpec:
    async def handler(tool_input: dict) -> dict:
        return {"tool": name}

    defaults = dict(
        name=name,
        description=f"tool {name}",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


def test_register_list_get() -> None:
    registry = ToolRegistry()
    spec_b = make_spec("beta")
    spec_a = make_spec("alpha")
    registry.register(spec_b)
    registry.register(spec_a)

    assert len(registry) == 2
    assert registry.get("alpha") is spec_a
    assert registry.get("missing") is None
    assert [s.name for s in registry.list()] == ["alpha", "beta"]
    assert registry.names() == ["alpha", "beta"]


def test_list_order_is_deterministic_regardless_of_registration_order() -> None:
    names = ["zeta", "alpha", "mango", "beta", "kiwi"]
    r1, r2 = ToolRegistry(), ToolRegistry()
    for n in names:
        r1.register(make_spec(n))
    for n in reversed(names):
        r2.register(make_spec(n))
    assert r1.api_schemas() == r2.api_schemas()
    assert [t["name"] for t in r1.api_schemas()] == sorted(names)


def test_api_schemas_shape() -> None:
    registry = ToolRegistry()
    registry.register(
        make_spec(
            "read_file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    )
    schemas = registry.api_schemas()
    assert schemas == [
        {
            "name": "read_file",
            "description": "tool read_file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry()
    registry.register(make_spec("dup"))
    with pytest.raises(ValueError, match="dup"):
        registry.register(make_spec("dup"))


def test_default_tools_are_safe_marks() -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    assert registry.names() == [
        "calculate",
        "delete_file",
        "read_file",
        "run_command",
        "search_workspace",
        "write_file",
    ]
    for name in ("calculate", "read_file", "search_workspace"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.dangerous is False
        assert spec.parallel_safe is True
        assert spec.access == "read"
    assert registry.get("write_file").access == "write"
    assert registry.get("delete_file").access == "write"
    assert registry.get("run_command").access == "execute"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "notes.txt").write_text("hello workspace", encoding="utf-8")
    (root / "sub" / "code.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return root


def tool_handler(registry: ToolRegistry, name: str, tool_input: dict) -> dict:
    spec = registry.get(name)
    assert spec is not None
    return asyncio.run(spec.handler(tool_input))


def test_read_file_inside_workspace(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    result = tool_handler(registry, "read_file", {"path": "notes.txt"})
    assert result["content"] == "hello workspace"
    assert result["path"] == "notes.txt"
    assert result["truncated"] is False

    nested = tool_handler(registry, "read_file", {"path": "sub/code.py"})
    assert "print" in nested["content"]


def test_read_file_rejects_parent_traversal(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    with pytest.raises(ToolPermanentError) as excinfo:
        tool_handler(registry, "read_file", {"path": "../secret.txt"})
    assert excinfo.value.code == "path_outside_workspace"


def test_read_file_rejects_absolute_path_outside(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    secret = workspace.parent / "secret.txt"
    with pytest.raises(ToolPermanentError) as excinfo:
        tool_handler(registry, "read_file", {"path": str(secret)})
    assert excinfo.value.code == "path_outside_workspace"


def test_read_file_rejects_deep_traversal_via_subdir(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    with pytest.raises(ToolPermanentError):
        tool_handler(registry, "read_file", {"path": "sub/../../secret.txt"})


def test_read_file_not_found_hint_lists_directory(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    with pytest.raises(ToolPermanentError) as excinfo:
        tool_handler(registry, "read_file", {"path": "nope.txt"})
    assert excinfo.value.code == "not_found"
    assert "notes.txt" in (excinfo.value.hint or "")


def test_read_file_validates_input(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)
    with pytest.raises(ToolInputError):
        tool_handler(registry, "read_file", {})


def test_search_workspace_finds_keyword(workspace: Path) -> None:
    (workspace / "sub" / "more.txt").write_text(
        "nothing\nfindme here\n", encoding="utf-8"
    )
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    result = tool_handler(registry, "search_workspace", {"pattern": "findme"})
    assert result["total"] == 1
    assert result["matches"][0]["file"] == "sub/more.txt"
    assert result["matches"][0]["line"] == 2


def test_search_workspace_skips_hidden_and_vendor_dirs(workspace: Path) -> None:
    junk = workspace / "node_modules" / "pkg"
    junk.mkdir(parents=True)
    (junk / "x.txt").write_text("findme in vendored", encoding="utf-8")
    (workspace / ".hidden.txt").write_text("findme hidden", encoding="utf-8")
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    result = tool_handler(registry, "search_workspace", {"pattern": "findme"})
    assert result["total"] == 0


def test_search_workspace_empty_pattern_invalid(workspace: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace)
    with pytest.raises(ToolInputError):
        tool_handler(registry, "search_workspace", {"pattern": ""})


def test_calculate_arithmetic() -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    assert tool_handler(registry, "calculate", {"expression": "1 + 2 * 3"})["result"] == 7
    assert tool_handler(registry, "calculate", {"expression": "(2 + 3) * 7 / 5"})["result"] == 7.0
    assert tool_handler(registry, "calculate", {"expression": "2 ** 10"})["result"] == 1024
    assert tool_handler(registry, "calculate", {"expression": "-5 // 2"})["result"] == -3


def test_calculate_rejects_code_injection() -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    for expr in [
        "__import__('os').system('ls')",
        "().__class__.__bases__",
        "eval('1+1')",
        "1; 2",
    ]:
        with pytest.raises(ToolInputError):
            tool_handler(registry, "calculate", {"expression": expr})


def test_calculate_zero_division_is_permanent_error() -> None:
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    with pytest.raises(ToolPermanentError) as excinfo:
        tool_handler(registry, "calculate", {"expression": "1 / 0"})
    assert excinfo.value.code == "zero_division"
