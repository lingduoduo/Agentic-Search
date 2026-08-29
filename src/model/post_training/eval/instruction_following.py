"""Objectively checkable instruction-following constraints.

Each predicate is decidable from the record alone, with no model in the loop.
That is the point: an LLM judge would make the headline comparison depend on a
third model's mood, and re-running the report a month later would produce a
different number for reasons unrelated to either policy.

The citation check deliberately requires more than a well-formed label. A model
that emits ``[R9Q9D9]`` having retrieved nothing has produced syntax, not a
citation, and rewarding it teaches exactly the wrong lesson.
"""

from __future__ import annotations

import json
import re

from .cohort import EvalRecord

CONSTRAINT_NAMES: tuple[str, ...] = (
    "answer_tag_present",
    "citations_wellformed",
    "tool_calls_parseable",
    "round_budget_respected",
)

# Mirrors the search-agent contract: <answer>...</answer> with a body.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*\S.*?</answer>", re.DOTALL | re.IGNORECASE)
# Mirrors src/context/search.py's citation labels.
_CITATION_RE = re.compile(r"\[(?:D\d+|R\d+Q\d+D\d+)\]")


def check_constraints(
    record: EvalRecord,
    *,
    allowed_tools: frozenset[str],
    max_search_rounds: int,
) -> dict[str, bool]:
    """Evaluate every constraint for one record.

    Always returns a verdict for every name in :data:`CONSTRAINT_NAMES`; a
    missing key would silently shrink a compliance rate's denominator.
    """
    labels = set(_CITATION_RE.findall(record.response))
    citations_ok = bool(labels) and all(
        label.strip("[]") in record.cited_ids for label in labels
    )

    tool_calls_ok = True
    for payload in record.tool_calls:
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            tool_calls_ok = False
            break
        if not isinstance(parsed, dict) or parsed.get("name") not in allowed_tools:
            tool_calls_ok = False
            break

    return {
        "answer_tag_present": bool(_ANSWER_TAG_RE.search(record.response)),
        "citations_wellformed": citations_ok,
        "tool_calls_parseable": tool_calls_ok,
        "round_budget_respected": (
            float(record.metrics.get("rounds_used", 0.0)) <= float(max_search_rounds)
        ),
    }
