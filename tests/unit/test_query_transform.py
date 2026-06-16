from __future__ import annotations

from src.context.query_transform import QueryTransformConfig, TransformedQueryBundle


def test_bundle_no_variants_returns_original():
    """When all expansions are empty, retrieval_variants returns [original]."""
    bundle = TransformedQueryBundle(original="what is FAISS?")
    assert bundle.retrieval_variants() == ["what is FAISS?"]


def test_retrieval_variants_deduplicates_case_insensitively():
    bundle = TransformedQueryBundle(
        original="what is FAISS?",
        sub_queries=["FAISS vector search", "faiss vector search", "FAISS indexing"],
    )
    variants = bundle.retrieval_variants()
    combined = [v.lower() for v in variants]
    assert combined.count("faiss vector search") == 1


def test_retrieval_variants_respects_max_variants():
    bundle = TransformedQueryBundle(
        original="q",
        sub_queries=[f"sub{i}" for i in range(10)],
    )
    assert len(bundle.retrieval_variants(max_variants=3)) == 3


def test_retrieval_variants_always_includes_original():
    """original must appear even when max_variants is tight."""
    bundle = TransformedQueryBundle(
        original="original query",
        sub_queries=["q1", "q2", "q3"],
    )
    variants = bundle.retrieval_variants(max_variants=2)
    assert len(variants) == 2
    assert "original query" in variants


def test_config_defaults_all_false():
    config = QueryTransformConfig()
    assert config.decompose is False
    assert config.hyde is False
    assert config.step_back is False
    assert config.keywords is False
    assert config.construct_filters is False
    assert config.max_variants == 5
