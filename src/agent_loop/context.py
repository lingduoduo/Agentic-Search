"""Data containers for multi-turn agentic search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    """One passage returned by a search or retrieval server."""

    contents: str
    score: float = 0.0
    title: str | None = None
    url: str | None = None

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
            title = doc.get("title")
            url = doc.get("url")
        else:
            contents = str(doc)
            title = None
            url = None

        return cls(contents=contents, score=score, title=title, url=url)


@dataclass
class SearchContext:
    """One query issued during a research round and the results returned."""

    query: str
    results: list[SearchResult] = field(default_factory=list)

    def to_information_block(self, citation_prefix: str | None = None) -> str:
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
            title = result.title or first_line.strip('"').strip()
            body = rest.strip() if rest.strip() else content
            label = f"[{citation_prefix}{i}] " if citation_prefix else ""
            suffix = f" URL: {result.url}" if result.url else ""
            if title:
                if citation_prefix:
                    lines.append(f"{label}(Title: {title}) {body}{suffix}")
                else:
                    lines.append(f"Doc {i}(Title: {title}) {body}{suffix}")
            else:
                if citation_prefix:
                    lines.append(f"{label}{content}{suffix}")
                else:
                    lines.append(f"Doc {i}: {content}{suffix}")
        return "\n".join(lines)


@dataclass
class AgentContext:
    """Accumulates all search turns performed during one agent run."""

    turns: list[SearchContext] = field(default_factory=list)
    rounds: list[list[SearchContext]] = field(default_factory=list)

    def add_turn(self, query: str, results: list[SearchResult]) -> SearchContext:
        return self.add_round([query], [results])[0]

    def add_round(
        self,
        queries: list[str],
        results_by_query: list[list[SearchResult]],
    ) -> list[SearchContext]:
        if len(queries) != len(results_by_query):
            raise ValueError("queries and results_by_query must have the same length.")
        round_contexts = [
            SearchContext(query=query, results=results)
            for query, results in zip(queries, results_by_query)
        ]
        self.turns.extend(round_contexts)
        self.rounds.append(round_contexts)
        return round_contexts

    @property
    def num_searches(self) -> int:
        return len(self.turns)

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def queries(self) -> list[str]:
        return [t.query for t in self.turns]
