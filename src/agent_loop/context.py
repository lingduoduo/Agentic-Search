"""Data containers for multi-turn agentic search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    """One passage returned by a search or retrieval server."""

    contents: str
    score: float = 0.0

    @classmethod
    def from_api_item(cls, item: dict[str, Any]) -> "SearchResult":
        """Parse either server response shape into a SearchResult.

        Retrieval server with return_scores=True:
            {"document": {"id": ..., "title": ..., "contents": ...}, "score": float}
        Google / SerpAPI servers (and retrieval without scores):
            {"document": {"contents": "\"Title\"\\nbody"}}
        Retrieval server with return_scores=False returns the document dict directly:
            {"id": ..., "title": ..., "contents": ...}
        """
        if "document" in item:
            doc = item["document"]
            score = float(item.get("score", 0.0))
        else:
            doc = item
            score = 0.0

        if isinstance(doc, dict):
            contents = doc.get("contents", str(doc))
        else:
            contents = str(doc)

        return cls(contents=contents, score=score)


@dataclass
class SearchContext:
    """One round of search: the query the model issued and the results returned."""

    query: str
    results: list[SearchResult] = field(default_factory=list)

    def to_information_block(self) -> str:
        """Format results as a plain-text block injected back into the conversation.

        Produces lines like:
            Doc 1(Title: Dense Retrieval with FAISS) Dense retrieval encodes...
        Falls back to the raw contents string when no title line is present.
        """
        if not self.results:
            return "No information available"

        lines: list[str] = []
        for i, result in enumerate(self.results, 1):
            content = result.contents
            first_line, _, rest = content.partition("\n")
            title = first_line.strip('"').strip()
            body = rest.strip() if rest.strip() else content
            if title:
                lines.append(f"Doc {i}(Title: {title}) {body}")
            else:
                lines.append(f"Doc {i}: {content}")
        return "\n".join(lines)


@dataclass
class AgentContext:
    """Accumulates all search turns performed during one agent run."""

    turns: list[SearchContext] = field(default_factory=list)

    def add_turn(self, query: str, results: list[SearchResult]) -> SearchContext:
        ctx = SearchContext(query=query, results=results)
        self.turns.append(ctx)
        return ctx

    @property
    def num_searches(self) -> int:
        return len(self.turns)

    @property
    def queries(self) -> list[str]:
        return [t.query for t in self.turns]
