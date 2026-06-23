"""SQL Query Generation — schema-aware Text-to-SQL with read-only validation.

No database is executed against. The LLM proposes SQL; validate_sql enforces
SELECT-only and a table/column allowlist before the query is returned.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.context.models import ChatMessage

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery

logger = logging.getLogger(__name__)

_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "merge",
    "replace",
    "attach",
    "pragma",
    ";--",
)


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[str, ...]


def _schema_text(schema: list[TableSchema]) -> str:
    return "\n".join(f"- {t.name}({', '.join(t.columns)})" for t in schema)


_SQL_PROMPT = """Translate the question into a single SQL SELECT query.
Use only these tables and columns:
{schema}
Rules: SELECT statements only. No INSERT/UPDATE/DELETE/DROP/CREATE. Return only the SQL.
Question: {query}
SQL:""".strip()


def validate_sql(sql: str, schema: list[TableSchema]) -> bool:
    """True iff sql is a single read-only SELECT over allowlisted tables.
    Substring-based: keyword checks may false-positive on those words inside
    string literals; CTE-shadowed names are not resolved (acceptable — this
    layer never executes SQL).
    """
    if not sql or not sql.strip():
        return False
    lowered = sql.lower()
    if not lowered.lstrip().startswith("select"):
        return False
    if any(word in lowered for word in _FORBIDDEN):
        return False
    if sql.count(";") > 1 or (";" in sql and not lowered.rstrip().endswith(";")):
        return False
    allowed_tables = {t.name.lower() for t in schema}
    # Allowlist tables referenced after FROM/JOIN.
    referenced_tables = re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    return all(tbl in allowed_tables for tbl in referenced_tables) and bool(
        referenced_tables
    )


class SqlQueryConstructor:
    def __init__(self, llm: object, schema: list[TableSchema]) -> None:
        self._llm = llm
        self._schema = schema

    def _generate(self, query: str) -> str:
        prompt = _SQL_PROMPT.format(schema=_schema_text(self._schema), query=query)
        resp = self._llm.complete([ChatMessage(role="user", content=prompt)])
        text = getattr(resp, "text", None) or str(resp)
        return text.strip().strip("`").removeprefix("sql").strip()

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        try:
            sql = self._generate(query)
        except Exception as exc:
            logger.warning("SQL generation failed: %s", exc)
            return ConstructedQuery(
                RetrieverTarget.SQL, {"sql": None, "error": "generation_failed"}, query
            )
        if not validate_sql(sql, self._schema):
            return ConstructedQuery(
                RetrieverTarget.SQL, {"sql": None, "error": "validation_failed"}, query
            )
        return ConstructedQuery(RetrieverTarget.SQL, {"sql": sql}, query)
