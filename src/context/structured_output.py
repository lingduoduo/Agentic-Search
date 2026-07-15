"""Provider-neutral contracts for schema-constrained completions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum


class StructuredOutputCapability(str, Enum):
    JSON_SCHEMA = "json_schema"
    PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class StructuredOutputRequest:
    name: str
    schema: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class StructuredCompletionMetadata:
    requested: bool = False
    applied: bool = False
    downgraded: bool = False
    refused: bool = False
    incomplete_reason: str | None = None


class SchemaUnsupportedError(RuntimeError):
    """The provider explicitly rejected JSON Schema output enforcement."""


_ANSWER_DRAFT_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims", "missing_information", "abstain"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
        },
        "abstain": {"type": "boolean"},
    },
}


def answer_draft_json_schema() -> dict[str, object]:
    """Return an independent copy of the canonical AnswerDraft JSON Schema."""
    return copy.deepcopy(_ANSWER_DRAFT_JSON_SCHEMA)
