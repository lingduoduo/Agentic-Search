"""Propose module labels for existing training examples.

Curating ~270 canonical examples is a review pass over machine proposals, not
270 acts of authorship. The cues below are the same ones
``src/internal/servers/web/intent_routing.py`` already routes on, which is why
the taxonomy has these fourteen modules and no others — so a proposal agrees
with the router by construction, and a disagreement is worth looking at.

Every proposal is a draft. The committed canonical file is the reviewed result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .intent_data import load_intent_examples
from .intent_taxonomy import modules_for_route

_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    "current_info": (
        re.compile(
            r"\b(latest|current|recent|news|price|stock|weather|today|now|"
            r"this week|right now)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_document": (
        re.compile(
            r"\b(doc|document|report|runbook|postmortem|checklist|spec|readme|"
            r"guide|notes|deck|spreadsheet|page|wiki|policy)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_fact": (
        re.compile(
            r"\b(which|who|when|where|how many|how much|what is the|value of|"
            r"number|setting|config|version)\b",
            re.IGNORECASE,
        ),
    ),
    "summarize": (re.compile(r"\b(summari[sz]e|tl;?dr|recap|overview of)\b", re.I),),
    "compare": (
        re.compile(r"\b(compare|versus|vs\.?|difference between|better than)\b", re.I),
    ),
    "generate": (
        re.compile(
            r"\b(write|draft|translate|rephrase|reword|rewrite|brainstorm|"
            r"compose|generate)\b",
            re.IGNORECASE,
        ),
    ),
    "converse": (
        re.compile(r"\b(hello|hi there|thanks|thank you|joke|poem|haiku)\b", re.I),
    ),
    "explain": (
        re.compile(
            r"\b(explain|why|how does|how do|what is|describe|tell me about)\b", re.I
        ),
    ),
    "create": (re.compile(r"\b(create|open|file|add|new)\b", re.IGNORECASE),),
    "send": (re.compile(r"\b(send|email|notify|post|message|share)\b", re.IGNORECASE),),
    "schedule": (
        re.compile(r"\b(schedule|book|remind|calendar|meeting|invite)\b", re.I),
    ),
    "modify": (
        re.compile(r"\b(update|change|delete|remove|cancel|close|rename|edit)\b", re.I),
    ),
    "execute": (
        re.compile(r"\b(run|execute|deploy|trigger|invoke|rerun|kick off)\b", re.I),
    ),
}

# Used when no cue fires, so every example still gets a valid starting label.
_DEFAULT_MODULE = {
    "search": "lookup_fact",
    "chat": "explain",
    "tool": "execute",
}


def propose_modules(text: str, route: str) -> tuple[str, ...]:
    """Propose one or more modules for *text* within *route*.

    Multi-label by design: "compare the current prices of BTC and ETH" is
    genuinely both a comparison and a request for current information.
    """
    candidates = tuple(
        module
        for module in modules_for_route(route)
        if module in _CUES and any(cue.search(text) for cue in _CUES[module])
    )
    return candidates or (_DEFAULT_MODULE[route],)


def write_seed_canonical(examples_path: Path, output_path: Path) -> int:
    """Write a proposed canonical file from labeled training examples."""
    examples = load_intent_examples(examples_path)
    records = [
        {
            "id": f"canon-{position:03d}",
            "text": example.text,
            "route": example.label,
            "modules": list(propose_modules(example.text, example.label)),
        }
        for position, example in enumerate(examples, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(records)
