"""Tests for Graph RAG entity extraction and retrieval expansion."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.graph_rag import (
    EntityGraph,
    build_entity_graph,
    extract_entities,
    graph_rag_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    doc_id: str, title: str = "", text: str = "", score: float = 1.0
) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title=title, text=text, url=None, score=score)


def _service(*result_batches: list[RetrievalResult]) -> MagicMock:
    """Return a mock service whose search() returns successive result batches."""
    svc = MagicMock()
    svc.search.side_effect = [(batch, "hybrid") for batch in result_batches]
    return svc


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_finds_multi_word_phrases():
    entities = extract_entities(
        "Dense Retrieval is powerful. Sparse Retrieval is fast."
    )
    assert "Dense Retrieval" in entities
    assert "Sparse Retrieval" in entities


def test_extract_entities_finds_acronyms():
    entities = extract_entities("FAISS and BM25 are widely used.")
    assert "FAISS" in entities
    assert "BM25" in entities


def test_extract_entities_deduplicates():
    entities = extract_entities("FAISS is fast. FAISS is popular.")
    assert entities.count("FAISS") == 1


def test_extract_entities_skips_stopwords():
    entities = extract_entities("The quick brown fox.")
    assert "The" not in entities


def test_extract_entities_skips_single_char_acronyms():
    entities = extract_entities("I use FAISS.")
    assert "I" not in entities
    assert "FAISS" in entities


def test_extract_entities_empty_text():
    assert extract_entities("") == []


def test_extract_entities_preserves_first_seen_order():
    entities = extract_entities("FAISS then BM25 then FAISS again.")
    assert entities.index("FAISS") < entities.index("BM25")


# ---------------------------------------------------------------------------
# EntityGraph
# ---------------------------------------------------------------------------


def test_entity_graph_add_document():
    g = EntityGraph()
    g.add_document("doc-1", ["FAISS", "Dense Retrieval"])
    assert "doc-1" in g.entity_docs.get("FAISS", set())
    assert "doc-1" in g.entity_docs.get("Dense Retrieval", set())


def test_entity_graph_neighbors_are_cooccurrences():
    g = EntityGraph()
    g.add_document("doc-1", ["FAISS", "ANN", "Dense Retrieval"])
    neighbors = g.neighbors("FAISS")
    assert "ANN" in neighbors
    assert "Dense Retrieval" in neighbors
    assert "FAISS" not in neighbors


def test_entity_graph_docs_for_entity():
    g = EntityGraph()
    g.add_document("doc-1", ["FAISS"])
    g.add_document("doc-2", ["FAISS", "BM25"])
    assert g.docs_for_entity("FAISS") == {"doc-1", "doc-2"}
    assert g.docs_for_entity("BM25") == {"doc-2"}


def test_entity_graph_unknown_entity_returns_empty():
    g = EntityGraph()
    assert g.neighbors("Unknown") == set()
    assert g.docs_for_entity("Unknown") == set()


# ---------------------------------------------------------------------------
# build_entity_graph
# ---------------------------------------------------------------------------


def test_build_entity_graph_uses_title_and_text():
    results = [_result("d1", title="Dense Retrieval", text="FAISS is used here.")]
    g = build_entity_graph(results)
    assert "Dense Retrieval" in g.doc_entities["d1"]
    assert "FAISS" in g.doc_entities["d1"]


def test_build_entity_graph_empty_results():
    g = build_entity_graph([])
    assert g.doc_entities == {}


# ---------------------------------------------------------------------------
# graph_rag_search
# ---------------------------------------------------------------------------


def test_graph_rag_search_returns_seed_results_when_no_expansion():
    seed = [_result("doc-1", title="FAISS", text="FAISS is a library.")]
    svc = _service(seed, [])  # entity search returns nothing
    results = graph_rag_search(
        "similarity search", service=svc, top_k=5, max_entity_queries=1
    )
    assert any(r.doc_id == "doc-1" for r in results)


def test_graph_rag_search_expands_with_entity_queries():
    seed = [_result("doc-1", title="FAISS", text="FAISS is a library.")]
    expansion = [
        _result("doc-2", title="ANN Search", text="Approximate Nearest Neighbor.")
    ]
    svc = _service(seed, expansion)
    results = graph_rag_search(
        "similarity search", service=svc, top_k=5, max_entity_queries=1
    )
    ids = {r.doc_id for r in results}
    assert "doc-1" in ids
    assert "doc-2" in ids


def test_graph_rag_search_respects_top_k():
    seed = [_result(f"seed-{i}", title="FAISS", text=f"Doc {i}.") for i in range(5)]
    expansion = [
        _result(f"exp-{i}", title="Dense Search", text=f"Exp {i}.") for i in range(5)
    ]
    svc = _service(seed, expansion)
    results = graph_rag_search(
        "dense retrieval", service=svc, top_k=3, max_entity_queries=1
    )
    assert len(results) <= 3


def test_graph_rag_search_empty_corpus_returns_empty():
    svc = MagicMock()
    svc.search.return_value = ([], "sparse_only")
    results = graph_rag_search("query", service=svc, top_k=5)
    assert results == []


def test_graph_rag_search_deduplicates_across_expansions():
    doc = _result("doc-1", title="FAISS", text="FAISS library.")
    svc = _service([doc], [doc])  # same doc in seed and expansion
    results = graph_rag_search("FAISS", service=svc, top_k=5, max_entity_queries=1)
    ids = [r.doc_id for r in results]
    assert ids.count("doc-1") == 1


def test_graph_rag_search_swallows_expansion_errors():
    seed = [_result("doc-1", title="Dense Retrieval", text="Dense Retrieval works.")]
    svc = MagicMock()
    svc.search.side_effect = [(seed, "hybrid"), RuntimeError("backend down")]
    results = graph_rag_search(
        "dense retrieval", service=svc, top_k=5, max_entity_queries=1
    )
    assert any(r.doc_id == "doc-1" for r in results)
