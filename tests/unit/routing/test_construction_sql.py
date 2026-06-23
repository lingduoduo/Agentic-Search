from src.internal.routing.construction.sql import (
    SqlQueryConstructor,
    TableSchema,
    validate_sql,
)
from src.internal.routing.route import RetrieverTarget, RouteDecision

_SCHEMA = [TableSchema("papers", ("id", "title", "year"))]


def _route():
    return RouteDecision(
        domain="structured",
        sources=["analytics_db"],
        retriever=RetrieverTarget.SQL,
        construction_target=RetrieverTarget.SQL,
    )


class _StubLLM:
    def __init__(self, sql):
        self._sql = sql

    def complete(self, messages, **kwargs):
        return self._sql


def test_validate_accepts_select_on_known_table():
    assert validate_sql("SELECT year, COUNT(*) FROM papers GROUP BY year", _SCHEMA)


def test_validate_rejects_non_select():
    assert not validate_sql("DROP TABLE papers", _SCHEMA)
    assert not validate_sql("DELETE FROM papers", _SCHEMA)


def test_validate_rejects_unknown_table():
    assert not validate_sql("SELECT * FROM users", _SCHEMA)


def test_constructor_returns_valid_sql():
    llm = _StubLLM("SELECT year, COUNT(*) FROM papers GROUP BY year")
    out = SqlQueryConstructor(llm, _SCHEMA).construct("papers per year", _route())
    assert out.target is RetrieverTarget.SQL
    assert out.payload["sql"].lower().startswith("select")


def test_constructor_rejects_invalid_sql():
    out = SqlQueryConstructor(_StubLLM("DROP TABLE papers"), _SCHEMA).construct(
        "delete everything", _route()
    )
    assert out.payload["sql"] is None
    assert out.payload["error"]


def test_constructor_degrades_on_llm_failure():
    class _Boom:
        def complete(self, messages, **kwargs):
            raise RuntimeError("no llm")

    out = SqlQueryConstructor(_Boom(), _SCHEMA).construct("x", _route())
    assert out.payload["sql"] is None


def test_validate_rejects_multi_statement():
    assert not validate_sql("SELECT 1; SELECT * FROM papers", _SCHEMA)


def test_validate_rejects_table_named_like_a_column():
    # 'year' is a column, not a table; FROM year must be rejected.
    assert not validate_sql("SELECT * FROM year", _SCHEMA)
