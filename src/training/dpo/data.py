"""Preference pairs: the input to DPO, and the loader that validates them.

Validation is strict, in the style of ``load_canonical_examples``: these pairs
*are* the training signal, so a silently skipped row changes what the model
learns with no later symptom. Every rejection names the line it came from,
because "invalid file" is useless against a thousand-line dataset.

This module imports nothing heavier than the standard library, so a caller can
inspect and validate a dataset without torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


@dataclass(frozen=True)
class PreferenceExample:
    """One preference pair: a prompt, a preferred response, a rejected one."""

    prompt: str
    chosen: str
    rejected: str


def load_preference_pairs(path: str | Path) -> list[PreferenceExample]:
    """Load and validate a JSONL file of ``{prompt, chosen, rejected}`` records.

    Blank lines are skipped — they are formatting, not corruption. Everything
    else that is not a complete, well-formed pair raises.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preference pair file is missing: {path}")

    pairs: list[PreferenceExample] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        pairs.append(_parse_line(raw, number, path))

    if not pairs:
        raise ValueError(f"Found no preference pairs in {path}")
    return pairs


def _parse_line(raw: str, number: int, path: Path) -> PreferenceExample:
    """Parse and validate one JSONL line into a ``PreferenceExample``."""
    where = f"{path}, line {number}"
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {where}: {exc.msg}") from exc

    if not isinstance(record, dict):
        raise ValueError(f"Record at {where} must be a JSON object")

    values = {}
    for field in _REQUIRED_FIELDS:
        value = record.get(field)
        if value is None:
            raise ValueError(f"Record at {where} has no {field!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Record at {where} has an empty {field!r}")
        values[field] = value

    if values["chosen"].strip() == values["rejected"].strip():
        # Contributes exactly log 2 of constant loss and zero gradient forever,
        # so it can never teach the model anything. It is also a common symptom
        # of a pair-construction bug upstream, which is worth surfacing loudly.
        raise ValueError(
            f"Record at {where} has identical 'chosen' and 'rejected' text; "
            "such a pair yields zero gradient and cannot train anything"
        )

    return PreferenceExample(**values)
