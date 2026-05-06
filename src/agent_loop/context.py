"""Data containers for multi-turn agentic search state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CITATION_RE = re.compile(r"\[R(\d+)Q(\d+)D(\d+)\]")


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
    task_id: str | None = None
    task_description: str | None = None

    def to_information_block(self, citation_prefix: str | None = None) -> str:
        """Format results as a plain-text block injected back into the conversation.

        Produces lines like:
            Doc 1(Title: Dense Retrieval with FAISS) Dense retrieval encodes...
        Falls back to the raw contents string when no title line is present.
        """
        if not self.results:
            return "No information available"

        lines: list[str] = []
        if self.task_id:
            task_line = f"Task {self.task_id}"
            if self.task_description:
                task_line += f": {self.task_description}"
            lines.append(task_line)
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
    tasks: dict[str, str] = field(default_factory=dict)
    fetched_pages: list[SearchResult] = field(default_factory=list)

    def register_tasks(self, tasks: dict[str, str]) -> None:
        self.tasks.update(tasks)

    def record_fetched_pages(self, pages: list[SearchResult]) -> None:
        self.fetched_pages.extend(pages)

    def add_turn(self, query: str, results: list[SearchResult]) -> SearchContext:
        return self.add_round([query], [results])[0]

    def add_round(
        self,
        queries: list[str],
        results_by_query: list[list[SearchResult]],
        task_ids: list[str | None] | None = None,
    ) -> list[SearchContext]:
        if len(queries) != len(results_by_query):
            raise ValueError("queries and results_by_query must have the same length.")
        task_ids = task_ids or [None] * len(queries)
        if len(task_ids) != len(queries):
            raise ValueError("task_ids and queries must have the same length.")
        round_contexts = [
            SearchContext(
                query=query,
                results=results,
                task_id=task_id,
                task_description=self.tasks.get(task_id) if task_id else None,
            )
            for query, results, task_id in zip(queries, results_by_query, task_ids)
        ]
        self.turns.extend(round_contexts)
        self.rounds.append(round_contexts)
        return round_contexts

    def cited_result_ids(self, answer_text: str) -> frozenset[str]:
        """Return the set of citation keys referenced in *answer_text*.

        Citation keys are formatted as ``R{round}Q{query}D{doc}`` (e.g. ``R1Q2D3``)
        by :meth:`SearchAgentLoop._format_round_information`.  Only keys that
        correspond to an actual retrieved result are included in the return value.
        """
        referenced = {
            f"R{r}Q{q}D{d}"
            for r, q, d in (
                (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                for m in _CITATION_RE.finditer(answer_text)
            )
        }
        valid: set[str] = set()
        for round_idx, round_ctxs in enumerate(self.rounds, 1):
            for query_idx, ctx in enumerate(round_ctxs, 1):
                for doc_idx in range(1, len(ctx.results) + 1):
                    key = f"R{round_idx}Q{query_idx}D{doc_idx}"
                    if key in referenced:
                        valid.add(key)
        return frozenset(valid)

    def cited_results(self, answer_text: str) -> list[SearchResult]:
        """Return retrieved results whose citation keys appear in *answer_text*."""
        cited_ids = self.cited_result_ids(answer_text)
        if not cited_ids:
            return []

        cited_results: list[SearchResult] = []
        for round_idx, round_ctxs in enumerate(self.rounds, 1):
            for query_idx, ctx in enumerate(round_ctxs, 1):
                for doc_idx, result in enumerate(ctx.results, 1):
                    key = f"R{round_idx}Q{query_idx}D{doc_idx}"
                    if key in cited_ids:
                        cited_results.append(result)
        return cited_results

    @property
    def num_results(self) -> int:
        """Total number of retrieved search results across all rounds."""
        return sum(len(ctx.results) for ctx in self.turns)

    @property
    def num_searches(self) -> int:
        return len(self.turns)

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def queries(self) -> list[str]:
        return [t.query for t in self.turns]
