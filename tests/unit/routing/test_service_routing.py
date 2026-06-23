from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.router import Router


class _StubBackend:
    def search_sparse(self, query, top_k, filters=None):
        return [RetrievalResult(doc_id="d1", title="t", text="x", url=None, score=1.0)]

    def search_dense(self, query, top_k, filters=None):
        raise NotImplementedError


def test_routing_disabled_runs_retrieval():
    svc = RetrievalService(_StubBackend())  # no router
    results, mode = svc.search("how many docs are there", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert not mode.startswith("routed:")


def test_routing_to_sql_short_circuits_to_empty():
    router = Router(RouteRegistry(DEFAULT_ROUTES))
    svc = RetrievalService(_StubBackend(), router=router)
    results, mode = svc.search("how many papers per year", top_k=3)
    assert results == []
    assert mode == "routed:sql"


def test_routing_to_hybrid_runs_retrieval():
    router = Router(RouteRegistry(DEFAULT_ROUTES))
    svc = RetrievalService(_StubBackend(), router=router)
    results, mode = svc.search("what is reciprocal rank fusion", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert not mode.startswith("routed:")
