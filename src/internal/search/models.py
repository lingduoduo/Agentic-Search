"""Normalized values passed between the internal search pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.context import ContextDocument
from src.context.search import SearchResult


@dataclass(frozen=True)
class CandidateSet:
    """Candidates returned by one retrieval provider for one query."""

    query: str
    candidates: list[SearchResult]
    provider: str
    filters: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "provider": self.provider,
            "filters": self.filters,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RankedEvidence:
    """Ordered, citation-addressable evidence ready for inference."""

    query: str
    evidence: list[ContextDocument]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for document in self.evidence:
            row = asdict(document)
            row["citation"] = document.citation
            rows.append(row)
        return {"query": self.query, "evidence": rows, "metadata": self.metadata}


@dataclass(frozen=True)
class GeneratedAnswer:
    """Text generated from evidence plus its normalized citation identifiers."""

    answer: str
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
