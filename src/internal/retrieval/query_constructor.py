"""LLM-based extraction of structured metadata filters from natural-language queries."""

from __future__ import annotations

import json
import logging

from src.context.models import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

_KNOWN_FILTER_FIELDS = frozenset(
    {"source", "date_year", "date_after", "date_before", "author", "doc_type"}
)

_EXTRACT_PROMPT = """Extract metadata filters from the user's query. Return JSON with exactly two keys:
- "query": the cleaned query with metadata phrases removed
- "filters": an object with any of these fields (omit fields not present):
  - "source": string (e.g. "arxiv", "confluence", "sharepoint")
  - "date_year": integer (e.g. 2023)
  - "date_after": string in "YYYY-MM-DD" format
  - "date_before": string in "YYYY-MM-DD" format
  - "author": string
  - "doc_type": string (e.g. "papers", "tickets", "pages")

Examples:
Query: "FAISS papers from 2023 on arxiv"
Output: {{"query": "FAISS papers", "filters": {{"date_year": 2023, "source": "arxiv"}}}}

Query: "what is attention mechanism"
Output: {{"query": "what is attention mechanism", "filters": {{}}}}

Query: {query}
Output:""".strip()


def _llm_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


class QueryConstructor:
    """Extract structured metadata filters from a natural-language query via LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract_filters(self, query: str) -> tuple[str, dict]:
        """Return (cleaned_query, filters).

        Falls back to (query, {}) on any LLM error or JSON parse failure — never raises.
        """
        try:
            raw = _llm_text(
                self._llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_EXTRACT_PROMPT.format(query=query)
                        )
                    ]
                )
            ).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].removeprefix("json").strip() if len(parts) > 1 else raw
            parsed = json.loads(raw)
            cleaned_query = str(parsed.get("query", query))
            raw_filters: dict = parsed.get("filters") or {}
            filters: dict = {
                k: v
                for k, v in raw_filters.items()
                if k in _KNOWN_FILTER_FIELDS and v is not None
            }
            return cleaned_query, filters
        except Exception as exc:
            logger.warning("Filter extraction failed: %s", exc)
            return query, {}
