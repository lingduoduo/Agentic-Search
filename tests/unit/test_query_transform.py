from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.context.query_transform import (
    QueryTransformConfig,
    QueryTransformPipeline,
    TransformedQueryBundle,
    config_signature,
)


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


def _llm_responses(responses: list[str]) -> MagicMock:
    """LLM mock returning responses in order for each complete() call."""
    m = MagicMock()
    m.complete.side_effect = responses
    return m


def test_pipeline_all_flags_off_returns_original_only():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = MagicMock()
    pipeline = QueryTransformPipeline(QueryTransformConfig(), llm)
    bundle = pipeline.transform("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert bundle.sub_queries == []
    assert bundle.hyde_text is None
    assert bundle.step_back is None
    assert bundle.keywords == []
    assert bundle.merged_filters == {}
    assert bundle.retrieval_variants() == ["what is FAISS?"]


def test_pipeline_decompose_flag_calls_enhancer():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = _llm_responses(["sub-q1\nsub-q2"])
    pipeline = QueryTransformPipeline(QueryTransformConfig(decompose=True), llm)
    bundle = pipeline.transform("compare FAISS and ScaNN")
    assert "sub-q1" in bundle.sub_queries
    assert "sub-q2" in bundle.sub_queries


def test_pipeline_hyde_flag_populates_hyde_text():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = _llm_responses(["FAISS is a fast library."])
    pipeline = QueryTransformPipeline(QueryTransformConfig(hyde=True), llm)
    bundle = pipeline.transform("what is FAISS?")
    assert bundle.hyde_text == "FAISS is a fast library."


def test_pipeline_step_back_flag_populates_step_back():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = _llm_responses(["What are vector similarity search algorithms?"])
    pipeline = QueryTransformPipeline(QueryTransformConfig(step_back=True), llm)
    bundle = pipeline.transform("how does FAISS GPU indexing work?")
    assert bundle.step_back == "What are vector similarity search algorithms?"


def test_pipeline_keywords_flag_calls_expand_keywords():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = MagicMock()
    with patch(
        "src.internal.servers.secondary_llm_flows.query_expansion.expand_keywords",
        return_value=["FAISS", "ANN index"],
    ) as mock_expand:
        pipeline = QueryTransformPipeline(QueryTransformConfig(keywords=True), llm)
        bundle = pipeline.transform("what is FAISS?")

    mock_expand.assert_called_once_with("what is FAISS?", llm)
    assert "FAISS" in bundle.keywords


def test_pipeline_construct_filters_merges_with_caller_filters():
    import json
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    extracted_payload = json.dumps(
        {"query": "FAISS papers", "filters": {"date_year": 2023, "source": "arxiv"}}
    )
    llm = _llm_responses([extracted_payload])
    caller_filters = {"source": "confluence"}  # caller wins on conflict
    pipeline = QueryTransformPipeline(QueryTransformConfig(construct_filters=True), llm)
    bundle = pipeline.transform(
        "FAISS papers from 2023 on arxiv", filters=caller_filters
    )
    assert bundle.merged_filters["source"] == "confluence"  # caller wins
    assert bundle.merged_filters["date_year"] == 2023


def test_from_env_returns_none_when_no_qt_vars_set(monkeypatch):
    from src.context.query_transform import QueryTransformPipeline

    for var in (
        "QT_DECOMPOSE",
        "QT_HYDE",
        "QT_STEP_BACK",
        "QT_KEYWORDS",
        "QT_CONSTRUCT_FILTERS",
    ):
        monkeypatch.delenv(var, raising=False)
    result = QueryTransformPipeline.from_env(MagicMock())
    assert result is None


def test_from_env_returns_pipeline_when_one_qt_var_set(monkeypatch):
    from src.context.query_transform import QueryTransformPipeline

    monkeypatch.setenv("QT_DECOMPOSE", "true")
    for var in ("QT_HYDE", "QT_STEP_BACK", "QT_KEYWORDS", "QT_CONSTRUCT_FILTERS"):
        monkeypatch.delenv(var, raising=False)
    pipeline = QueryTransformPipeline.from_env(MagicMock())
    assert pipeline is not None
    assert pipeline._config.decompose is True
    assert pipeline._config.hyde is False


def test_retrieval_variants_includes_original_when_it_appears_in_sub_queries():
    """original must still appear (last) even if it was also in sub_queries."""
    bundle = TransformedQueryBundle(
        original="what is FAISS?",
        sub_queries=["what is FAISS?", "q1", "q2"],
    )
    variants = bundle.retrieval_variants(max_variants=3)
    assert variants[-1] == "what is FAISS?"
    # sub-queries that aren't original fill the other slots
    assert len(variants) == 3
    assert variants.count("what is FAISS?") == 1


# --- Task 1: job-based refactor tests ---


def _fake_llm(text: str = "") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_transform_config_override_runs_only_overridden_transforms():
    # Leaf built with everything OFF; override turns step_back ON.
    pipe = QueryTransformPipeline(QueryTransformConfig(), _fake_llm("broader query"))
    override = QueryTransformConfig(step_back=True)
    bundle = pipe.transform("specific q", config_override=override)
    assert bundle.step_back == "broader query"
    assert bundle.sub_queries == []  # decompose stayed off


def test_base_config_exposed():
    cfg = QueryTransformConfig(hyde=True)
    pipe = QueryTransformPipeline(cfg, _fake_llm())
    assert pipe.base_config is cfg


def test_config_signature_changes_with_flags():
    a = config_signature(QueryTransformConfig(hyde=True))
    b = config_signature(QueryTransformConfig(hyde=False))
    assert a != b
