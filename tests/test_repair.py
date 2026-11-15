
from __future__ import annotations

import copy

from app.core.repair import (
    SYNTHETIC_INTERRUPTED,
    find_orphans,
    needs_repair,
    repair_orphans,
)


def user_text(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def user_results(*tool_use_ids: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": "ok", "is_error": False}
            for tid in tool_use_ids
        ],
    }


def assistant_with(*tool_use_ids: str, text: str = "") -> dict:
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(
        {"type": "tool_use", "id": tid, "name": "echo", "input": {}} for tid in tool_use_ids
    )
    return {"role": "assistant", "content": content}


def test_detects_orphan_tool_use() -> None:
    messages = [user_text("hi"), assistant_with("tu_1")]
    assert needs_repair(messages) is True
    assert find_orphans(messages) == [(1, "tu_1")]


def test_paired_tool_use_is_not_orphan() -> None:
    messages = [user_text("hi"), assistant_with("tu_1"), user_results("tu_1")]
    assert needs_repair(messages) is False
    assert find_orphans(messages) == []


def test_repair_makes_list_orphan_free_and_is_idempotent() -> None:
    messages = [user_text("hi"), assistant_with("tu_1", text="working")]
    repaired = repair_orphans(messages)

    assert needs_repair(repaired) is False

    assert len(repaired) == 3
    assert repaired[2]["role"] == "user"
    block = repaired[2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_1"
    assert block["content"] == SYNTHETIC_INTERRUPTED
    assert block["is_error"] is True

    assert repair_orphans(repaired) == repaired


def test_multiple_orphans_merged_into_one_synthetic_message() -> None:
    messages = [assistant_with("tu_a", "tu_b", "tu_c")]
    assert find_orphans(messages) == [(0, "tu_a"), (0, "tu_b"), (0, "tu_c")]

    repaired = repair_orphans(messages)
    assert len(repaired) == 2
    synthetic = repaired[1]
    assert synthetic["role"] == "user"
    assert [b["tool_use_id"] for b in synthetic["content"]] == ["tu_a", "tu_b", "tu_c"]
    assert needs_repair(repaired) is False


def test_orphans_across_multiple_assistant_messages() -> None:
    messages = [
        assistant_with("tu_1"),
        assistant_with("tu_2"),
    ]
    repaired = repair_orphans(messages)
    assert [m["role"] for m in repaired] == ["assistant", "user", "assistant", "user"]
    assert repaired[1]["content"][0]["tool_use_id"] == "tu_1"
    assert repaired[3]["content"][0]["tool_use_id"] == "tu_2"
    assert needs_repair(repaired) is False


def test_mixed_normal_messages_untouched() -> None:
    messages = [
        user_text("q1"),
        assistant_with("tu_ok", text="checking"),
        user_results("tu_ok"),
        assistant_with("tu_lost", text="hmm"),
        user_text("q2"),
    ]
    repaired = repair_orphans(messages)

    assert needs_repair(repaired) is False
    assert len(repaired) == 6
    assert repaired[2] == user_results("tu_ok")
    assert repaired[4]["content"][0]["tool_use_id"] == "tu_lost"
    assert repaired[5] == user_text("q2")


def test_repair_does_not_mutate_input() -> None:
    messages = [assistant_with("tu_1")]
    snapshot = copy.deepcopy(messages)
    repaired = repair_orphans(messages)
    assert messages == snapshot
    assert repaired is not messages


def test_partial_pairing_only_repairs_missing_ones() -> None:
    messages = [
        assistant_with("tu_1", "tu_2"),
        user_results("tu_1"),
    ]
    assert find_orphans(messages) == [(0, "tu_2")]
    repaired = repair_orphans(messages)
    assert needs_repair(repaired) is False
    assert repaired[1]["content"][0]["tool_use_id"] == "tu_2"
    assert repaired[1]["content"][0]["is_error"] is True
    assert repaired[2] == user_results("tu_1")


def test_content_string_messages_are_skipped() -> None:
    messages = [
        {"role": "user", "content": "plain text"},
        {"role": "assistant", "content": "plain reply"},
    ]
    assert needs_repair(messages) is False
    assert repair_orphans(messages) == messages
