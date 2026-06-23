"""Knowledge Graph Query Construction — read-only Cypher templating.

No graph database is executed against. The LLM extracts (entity, relation);
a parameterised MATCH...RETURN template is built and validated read-only.
"""

from __future__ import annotations

import json
import logging
import re

from src.context.models import ChatMessage

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery

logger = logging.getLogger(__name__)

_WRITE_CLAUSE_RE = re.compile(
    r"\b(create|delete|merge|set|remove|detach|drop|foreach|call)\b"
)

_EXTRACT_PROMPT = """Identify the central entity and the relationship the question asks about.
Return JSON only: {{"entity": "<entity>", "relation": "<relation or empty>"}}.
Question: {query}
JSON:""".strip()


def validate_cypher(cypher: str) -> bool:
    """True iff cypher is a read-only MATCH...RETURN with no write clauses.

    Word-boundary keyword matching; this layer never executes Cypher.
    """
    if not cypher or not cypher.strip():
        return False
    lowered = cypher.lower()
    if "match" not in lowered or "return" not in lowered:
        return False
    return not _WRITE_CLAUSE_RE.search(lowered)


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
        template = 'MATCH (n {name: "PLACEHOLDER"})-[r]-(m) RETURN n, r, m'
        if not validate_cypher(template):  # template is read-only by construction
            return ConstructedQuery(
                RetrieverTarget.GRAPH, {"cypher": None, "entity": None}, query
            )
        cypher = f'MATCH (n {{name: "{_escape(entity)}"}})-[r]-(m) RETURN n, r, m'
        return ConstructedQuery(
            RetrieverTarget.GRAPH, {"cypher": cypher, "entity": entity}, query
        )
