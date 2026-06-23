from __future__ import annotations

from src.internal.routing.construction.base import ConstructedQuery
from src.internal.routing.construction.hybrid import HybridRetrievalQueryConstructor
from src.internal.routing.construction.metadata import MetadataFilterConstructor
from src.internal.routing.construction.vector import VectorSearchQueryConstructor
from src.internal.routing.route import RetrieverTarget, RouteDecision


def _route(target):
    return RouteDecision(
        domain="docs", sources=["local"], retriever=target, construction_target=target
    )


class _StubLLM:
    def complete(self, messages, **kwargs):
        return '{"query": "faiss papers", "filters": {"date_year": 2023}}'


def test_metadata_constructor_extracts_filters():
    c = MetadataFilterConstructor(_StubLLM())
    out = c.construct("faiss papers from 2023", _route(RetrieverTarget.METADATA))
    assert isinstance(out, ConstructedQuery)
    assert out.target is RetrieverTarget.METADATA
    assert out.payload["filters"] == {"date_year": 2023}
    assert out.text == "faiss papers"


def test_metadata_constructor_degrades_without_llm():
    class _Boom:
        def complete(self, messages, **kwargs):
            raise RuntimeError("no llm")

    c = MetadataFilterConstructor(_Boom())
    out = c.construct("anything", _route(RetrieverTarget.METADATA))
    assert out.payload["filters"] == {}
    assert out.text == "anything"


def test_vector_constructor_carries_params():
    c = VectorSearchQueryConstructor(top_k=8)
    out = c.construct("dense search please", _route(RetrieverTarget.DENSE))
    assert out.target is RetrieverTarget.DENSE
    assert out.payload["top_k"] == 8
    assert out.payload["namespace"] == "local"
    assert out.text == "dense search please"


def test_hybrid_constructor_sets_adaptive_lambda():
    c = HybridRetrievalQueryConstructor(rrf_k=60)
    out = c.construct("faiss", _route(RetrieverTarget.HYBRID))  # 1 token → lambda 0.8
    assert out.target is RetrieverTarget.HYBRID
    assert out.payload["rrf_k"] == 60
    assert out.payload["mmr_lambda"] == 0.8
    assert out.payload["w_sparse"] == 0.5
