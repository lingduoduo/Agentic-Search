"""Knowledge Graph Query Construction — read-only Cypher templating.

No graph database is executed against. The LLM extracts (entity, relation);
a parameterised MATCH...RETURN template is built and validated read-only.
"""

from __future__ import annotations

import json
import logging

from src.context.models import ChatMessage

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery

logger = logging.getLogger(__name__)

_WRITE_CLAUSES = ("create", "delete", "merge", "set ", "remove", "detach", "drop")

_EXTRACT_PROMPT = """Identify the central entity and the relationship the question asks about.
Return JSON only: {{"entity": "<entity>", "relation": "<relation or empty>"}}.
Question: {query}
JSON:""".strip()


def validate_cypher(cypher: str) -> bool:
    """True iff cypher is a read-only MATCH...RETURN with no write clauses."""
    if not cypher or not cypher.strip():
        return False
    lowered = cypher.lower()
    if "match" not in lowered or "return" not in lowered:
        return False
    return not any(clause in lowered for clause in _WRITE_CLAUSES)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class KnowledgeGraphQueryConstructor:
    def __init__(self, llm: object) -> None:
        self._llm = llm

    def _extract_entity(self, query: str) -> str | None:
        resp = self._llm.complete(
            [ChatMessage(role="user", content=_EXTRACT_PROMPT.format(query=query))]
        )
        text = (getattr(resp, "text", None) or str(resp)).strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        entity = str(data.get("entity", "")).strip()
        return entity or None

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        try:
            entity = self._extract_entity(query)
        except Exception as exc:
            logger.warning("KG entity extraction failed: %s", exc)
            entity = None
        if not entity:
            return ConstructedQuery(
                RetrieverTarget.GRAPH, {"cypher": None, "entity": None}, query
            )
        cypher = f'MATCH (n {{name: "{_escape(entity)}"}})-[r]-(m) RETURN n, r, m'
        if not validate_cypher(cypher):  # defensive; template is read-only by design
            return ConstructedQuery(
                RetrieverTarget.GRAPH, {"cypher": None, "entity": entity}, query
            )
        return ConstructedQuery(
            RetrieverTarget.GRAPH, {"cypher": cypher, "entity": entity}, query
        )
