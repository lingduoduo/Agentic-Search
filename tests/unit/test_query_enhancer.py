from __future__ import annotations

from unittest.mock import MagicMock

from src.context.query_enhancer import QueryBundle, QueryEnhancer


def _llm(response: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# QueryBundle
# ---------------------------------------------------------------------------


def test_query_bundle_all_queries_deduped():
    b = QueryBundle(
        original="what is FAISS?",
        sub_queries=[
            "what is FAISS?",
            "FAISS vector search",
            "facebook AI similarity search",
        ],
        hyde_text="FAISS is a library for efficient similarity search developed by Facebook AI.",
    )
    queries = b.all_queries()
    assert queries[0] == "what is FAISS?"
    assert "FAISS vector search" in queries
    assert "facebook AI similarity search" in queries
    assert "FAISS is a library" in queries[-1]
    # original not duplicated
    assert queries.count("what is FAISS?") == 1


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def test_decompose_returns_sub_queries():
    llm = _llm(
        "FAISS vector search\nfacebook AI similarity search\nnearest neighbour index"
    )
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose("what is FAISS and how does it compare to ScaNN?")
    assert len(result) >= 1
    assert all(isinstance(q, str) and q for q in result)


def test_decompose_excludes_original():
    query = "what is FAISS?"
    llm = _llm(f"{query}\nFAISS vector search")
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose(query)
    assert query not in result


def test_decompose_returns_original_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


def test_decompose_returns_original_on_llm_failure():
    llm = _llm.__wrapped__ if hasattr(_llm, "__wrapped__") else MagicMock()
    llm.complete.side_effect = RuntimeError("LLM unavailable")
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


def test_decompose_returns_original_on_empty_response():
    enhancer = QueryEnhancer(_llm(""))
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


# ---------------------------------------------------------------------------
# hyde
# ---------------------------------------------------------------------------


def test_hyde_returns_hypothetical_text():
    hypothetical = "FAISS is a library developed by Facebook AI Research for efficient similarity search."
    enhancer = QueryEnhancer(_llm(hypothetical))
    result = enhancer.hyde("what is FAISS?")
    assert result == hypothetical


def test_hyde_returns_none_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    assert enhancer.hyde("what is FAISS?") is None


def test_hyde_returns_none_on_llm_failure():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("timeout")
    enhancer = QueryEnhancer(llm)
    assert enhancer.hyde("what is FAISS?") is None


# ---------------------------------------------------------------------------
# enhance
# ---------------------------------------------------------------------------


def test_enhance_returns_bundle():
    llm = MagicMock()
    llm.complete.side_effect = [
        "FAISS vector search\nFacebook similarity search",  # decompose call
        "FAISS is a fast similarity search library.",  # hyde call
    ]
    enhancer = QueryEnhancer(llm)
    bundle = enhancer.enhance("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert len(bundle.sub_queries) >= 1
    assert bundle.hyde_text is not None


def test_enhance_no_llm_returns_passthrough_bundle():
    enhancer = QueryEnhancer(llm=None)
    bundle = enhancer.enhance("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert bundle.sub_queries == ["what is FAISS?"]
    assert bundle.hyde_text is None
