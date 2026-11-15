"""孤儿 tool_use 修复（不变量 1：无孤儿 tool_use）。

API 强制要求每个 tool_use 块有配对 tool_result，缺一个下次请求直接 400。
本模块对消息数组做纯函数式扫描与修复：

- needs_repair(messages) -> bool：是否存在无配对 tool_result 的 tool_use
- repair_orphans(messages) -> list：返回修复后的新列表——在每个含孤儿
  tool_use 的 assistant 消息之后插入一条合成 user 消息：

      {"type": "tool_result",
       "tool_use_id": <id>,
       "content": "interrupted: process crashed before this tool executed",
       "is_error": True}

服务启动时对 DB 中所有未完成会话执行一遍（main.lifespan，P1 下一波次接入）；
AgentLoop 每轮结束前也用它做防御性校验。

消息形态为 API 格式 dict：{"role": "user"|"assistant", "content": [blocks]}。
content 为纯字符串的消息不可能含 tool_use，直接跳过。
"""

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
    """扫描孤儿 tool_use，返回 [(消息下标, tool_use_id), ...]。

    配对规则：tool_result 允许出现在其后任意一条 user 消息中（本实现按
    「第一个包含该 tool_use_id 的 user 消息」匹配，覆盖正常紧邻配对）。
    """
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
    """返回修复后的新列表（不修改入参）。

    每个含孤儿 tool_use 的 assistant 消息后插入一条合成 user 消息，
    携带该消息全部孤儿对应的 tool_result（多孤儿合并进同一条消息）。
    幂等：对已修复列表再次调用为无操作。
    """
    orphans = find_orphans(messages)
    if not orphans:
        return [dict(m) for m in messages]

    orphans_by_index: dict[int, list[str]] = {}
    for index, tool_use_id in orphans:
        orphans_by_index.setdefault(index, []).append(tool_use_id)

    repaired: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        # 浅拷贝消息 dict，content 列表原样共享（本函数不修改任何既有块）
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
