"""Conversation-memory service: MCP-free logic over AgenticSearchStore."""

from __future__ import annotations

import re
from typing import Any, Callable

from src.internal.db.models import UserMemoryRecord

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
