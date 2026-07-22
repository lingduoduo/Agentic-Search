"""Conversation-memory service: MCP-free logic over AgenticSearchStore."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable

from src.internal.db.models import UserMemoryRecord
from src.internal.memory.tools import build_memory_registry

DEFAULT_MEMORY_USER_ID = "default_user"
MAX_CURATION_TURNS = 6
MEMORY_GATHER_CHAR_BUDGET = 12000

Encoder = Callable[[list[str]], Any]  # list[str] -> np.ndarray


def save_memory(store, user_id: str, text: str) -> str | None:
    record = store.add_user_memory(user_id, text)
    return record.id if record is not None else None


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if t]


def search_memories(
    store,
    user_id: str,
    query: str,
    max_results: int = 5,
    encoder: Encoder | None = None,
) -> list[tuple[UserMemoryRecord, float]]:
    records = store.get_user_memory_records(user_id)
    if not records or not query.strip():
        return []
    if encoder is not None:
        import numpy as np

        matrix = encoder([f"passage: {r.memory_text}" for r in records])
        qvec = encoder([f"query: {query}"])
        sims = (np.asarray(qvec) @ np.asarray(matrix).T)[0]
        ranked = sorted(zip(records, sims), key=lambda x: float(x[1]), reverse=True)
        return [(r, float(s)) for r, s in ranked[:max_results]]
    # Lexical fallback: token-overlap, normalized by query length.
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []
    scored: list[tuple[UserMemoryRecord, float]] = []
    for r in records:
        overlap = len(q_tokens & set(_tokenize(r.memory_text)))
        if overlap:
            scored.append((r, overlap / len(q_tokens)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_results]


def _attribute(record: UserMemoryRecord) -> str | None:
    tags = record.metadata.get("tags") if isinstance(record.metadata, dict) else None
    if isinstance(tags, list) and tags and isinstance(tags[0], str) and tags[0].strip():
        return tags[0].strip()
    return None


def consolidate_memories(
    store, user_id: str, resolve_conflicts: bool = True
) -> dict[str, Any]:
    records = store.get_user_memory_records(user_id)
    report: dict[str, Any] = {
        "initial": len(records),
        "duplicates_removed": 0,
        "conflicts_resolved": [],
        "final": 0,
    }

    def newest_first(rs: list[UserMemoryRecord]) -> list[UserMemoryRecord]:
        return sorted(rs, key=lambda r: (r.updated_at or "", r.id), reverse=True)

    # Exact-content dedup, keeping the newest.
    seen: set[str] = set()
    keep: list[UserMemoryRecord] = []
    remove: list[UserMemoryRecord] = []
    for r in newest_first(records):
        key = r.memory_text.strip().lower()
        if key in seen:
            remove.append(r)
            report["duplicates_removed"] += 1
            continue
        seen.add(key)
        keep.append(r)

    if resolve_conflicts:
        by_attr: dict[str, list[UserMemoryRecord]] = {}
        untagged: list[UserMemoryRecord] = []
        for r in keep:
            attr = _attribute(r)
            (untagged if attr is None else by_attr.setdefault(attr, [])).append(r)
        resolved = list(untagged)
        for attr, group in by_attr.items():
            ordered = newest_first(group)
            resolved.append(ordered[0])
            if len(ordered) > 1:
                remove.extend(ordered[1:])
                report["conflicts_resolved"].append(
                    {
                        "attribute": attr,
                        "kept": ordered[0].memory_text,
                        "superseded": [r.memory_text for r in ordered[1:]],
                    }
                )
        keep = resolved

    for r in remove:
        store.delete_user_memory(user_id, r.id)
    report["final"] = len(keep)
    return report


_CURATION_SYSTEM = (
    "You maintain a user's long-term memory. Given a recent conversation and the "
    "user's current memories, reconcile them by calling add_memory, update_memory, "
    "and delete_memory. Add new durable facts/preferences, update changed ones, and "
    "delete outdated or contradicted ones. Keep each memory a single contextual "
    "sentence. Do NOT store secrets (passwords, PINs, full SSNs, or full card/account "
    "numbers). When there is nothing left to change, reply with STOP and no tool calls."
)

_CURATION_USER = (
    "Recent conversation:\n{conversation}\n\n"
    "Current memories (id: text):\n{memories}\n\n"
    "Update the memory set now using the tools."
)


def _gather_sources(store, user_id: str, session_id: str | None) -> str:
    sessions = (
        [store.get_chat_session(session_id)]
        if session_id
        else store.list_sessions_for_user(user_id)
    )
    lines: list[str] = []
    for sess in sessions:
        if sess is None:
            continue
        for msg in store.list_chat_messages(sess.id):
            lines.append(f"{msg.role.upper()}: {msg.content}")
    text = "\n".join(lines)
    return text[-MEMORY_GATHER_CHAR_BUDGET:]


def _format_memories(store, user_id: str) -> str:
    records = store.get_user_memory_records(user_id)
    if not records:
        return "(none)"
    return "\n".join(f"{r.id}: {r.memory_text}" for r in records)


def _stream_turn(
    llm, messages: list[dict], schemas: list[dict]
) -> tuple[str, list[dict]]:
    content_parts: list[str] = []
    acc: dict[int, dict[str, str]] = {}
    for chunk in llm.stream(messages, tools=schemas, max_tokens=1024):
        delta = chunk.choice.delta
        if delta.content:
            content_parts.append(delta.content)
        for tcd in delta.tool_calls:
            slot = acc.setdefault(tcd.index, {"id": "", "name": "", "arguments": ""})
            if tcd.id:
                slot["id"] = tcd.id
            if tcd.function is not None:
                if tcd.function.name:
                    slot["name"] = tcd.function.name
                if tcd.function.arguments:
                    slot["arguments"] += tcd.function.arguments
    tool_calls = [acc[i] for i in sorted(acc)]
    return "".join(content_parts), tool_calls


async def curate_from_conversation(
    store,
    user_id: str,
    llm,
    session_id: str | None = None,
    max_turns: int = MAX_CURATION_TURNS,
) -> dict[str, Any]:
    sources = _gather_sources(store, user_id, session_id)
    if not sources.strip():
        return {
            "status": "empty",
            "message": "no conversations or notes yet",
            "counts": {},
        }

    before = [r.memory_text for r in store.get_user_memory_records(user_id)]
    registry, counts, schemas = build_memory_registry(store, user_id)
    messages: list[dict] = [
        {"role": "system", "content": _CURATION_SYSTEM},
        {
            "role": "user",
            "content": _CURATION_USER.format(
                conversation=sources, memories=_format_memories(store, user_id)
            ),
        },
    ]
    tool_call_log: list[dict] = []
    for _ in range(max_turns):
        content, tool_calls = await asyncio.to_thread(
            _stream_turn, llm, messages, schemas
        )
        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
        messages.append(assistant)
        if not tool_calls:
            break
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                result = f"error: invalid JSON arguments: {exc}"
            else:
                response, _raw, errors = await registry.invoke(tc["name"], args)
                result = response or ("; ".join(errors) if errors else "ok")
                tool_call_log.append(
                    {"name": tc["name"], "arguments": args, "result": result}
                )
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    after = [r.memory_text for r in store.get_user_memory_records(user_id)]
    trajectory = {
        "memory_before": before,
        "tool_calls": tool_call_log,
        "memory_after": after,
        "counts": dict(counts),
    }
    record = store.add_memory_trajectory(
        user_id,
        session_id=session_id,
        model=getattr(getattr(llm, "config", None), "model_name", ""),
        trajectory=trajectory,
    )
    return {
        "status": "ok",
        "trajectory_id": record.id,
        "counts": dict(counts),
        "memory_count": len(after),
    }
