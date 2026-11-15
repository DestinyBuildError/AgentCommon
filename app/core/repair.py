
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SYNTHETIC_INTERRUPTED = "interrupted: process crashed before this tool executed"


def _blocks(message: Mapping[str, Any]) -> list[Any]:
    content = message.get("content")
    if isinstance(content, list):
        return content
    return []


def _tool_use_ids(blocks: Sequence[Any]) -> list[str]:
    return [b.get("id") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")]


def _tool_result_ids(blocks: Sequence[Any]) -> set[str]:
    return {
        b.get("tool_use_id")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id")
    }


def find_orphans(messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, str]]:
    answered: set[str] = set()
    for message in messages:
        if message.get("role") == "user":
            answered |= _tool_result_ids(_blocks(message))

    orphans: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for tool_use_id in _tool_use_ids(_blocks(message)):
            if tool_use_id in answered or tool_use_id in seen:
                continue
            seen.add(tool_use_id)
            orphans.append((index, tool_use_id))
    return orphans


def needs_repair(messages: Sequence[Mapping[str, Any]]) -> bool:
    return bool(find_orphans(messages))


def synthetic_tool_result(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": SYNTHETIC_INTERRUPTED,
        "is_error": True,
    }


def repair_orphans(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    orphans = find_orphans(messages)
    if not orphans:
        return [dict(m) for m in messages]

    orphans_by_index: dict[int, list[str]] = {}
    for index, tool_use_id in orphans:
        orphans_by_index.setdefault(index, []).append(tool_use_id)

    repaired: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, list):
            copy["content"] = list(content)
        repaired.append(copy)

        orphan_ids = orphans_by_index.get(index)
        if orphan_ids:
            repaired.append(
                {
                    "role": "user",
                    "content": [synthetic_tool_result(tid) for tid in orphan_ids],
                }
            )
    return repaired
