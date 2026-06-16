from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.internal.retrieval.query_constructor import QueryConstructor


def _llm(response: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    return m


def test_extracts_date_year():
    payload = json.dumps({"query": "FAISS papers", "filters": {"date_year": 2023}})
    constructor = QueryConstructor(_llm(payload))
    cleaned, filters = constructor.extract_filters("FAISS papers from 2023")
    assert filters.get("date_year") == 2023
    assert cleaned == "FAISS papers"


def test_extracts_source():
    payload = json.dumps({"query": "dense retrieval", "filters": {"source": "arxiv"}})
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("arxiv articles about dense retrieval")
    assert filters.get("source") == "arxiv"


def test_extracts_multiple_fields():
    payload = json.dumps(
        {
            "query": "Hinton papers",
            "filters": {
                "author": "Hinton",
                "source": "arxiv",
                "date_before": "2024-01-01",
            },
        }
    )
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("Hinton papers on arxiv before 2024")
    assert filters["author"] == "Hinton"
    assert filters["source"] == "arxiv"
    assert filters["date_before"] == "2024-01-01"


def test_fallback_on_malformed_json():
    """Invalid JSON from LLM → (original_query, {}) with no exception."""
    constructor = QueryConstructor(_llm("not valid json {{"))
    cleaned, filters = constructor.extract_filters("FAISS papers from 2023")
    assert cleaned == "FAISS papers from 2023"
    assert filters == {}


def test_fallback_on_llm_error():
    """LLM exception → (original_query, {}) with no exception."""
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("LLM down")
    constructor = QueryConstructor(llm)
    cleaned, filters = constructor.extract_filters("any query")
    assert cleaned == "any query"
    assert filters == {}


def test_unknown_filter_fields_dropped():
    payload = json.dumps(
        {
            "query": "papers",
            "filters": {
                "source": "arxiv",
                "unknown_field": "value",
                "another_unknown": 42,
            },
        }
    )
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("papers")
    assert "unknown_field" not in filters
    assert "another_unknown" not in filters
    assert filters["source"] == "arxiv"
