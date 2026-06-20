from __future__ import annotations

from unittest.mock import MagicMock

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.query_router import QueryRouter
from src.internal.retrieval.routed_query_transform import RoutedQueryTransformPipeline


def _llm(text="broad"):
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_router_config_applied_per_query():
    # Leaf built with all-off; router forces step_back on for a question query.
    leaf = QueryTransformPipeline(QueryTransformConfig(), _llm("broad"))
    routed = RoutedQueryTransformPipeline(leaf, QueryRouter(model_path=None))
    bundle = routed.transform("why does HNSW work")
    assert bundle.step_back == "broad"  # heuristic enabled step_back


def test_explicit_override_wins_over_router():
    leaf = QueryTransformPipeline(QueryTransformConfig(), _llm("broad"))
    routed = RoutedQueryTransformPipeline(leaf, QueryRouter(model_path=None))
    bundle = routed.transform("why x", config_override=QueryTransformConfig())
    assert bundle.step_back is None  # forced all-off override respected
