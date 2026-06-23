from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.route import RetrieverTarget
from src.internal.routing.router import Router


def _router():
    return Router(RouteRegistry(DEFAULT_ROUTES))


def test_aggregation_query_routes_to_sql():
    d = _router().route("how many papers were published per year")
    assert d.retriever is RetrieverTarget.SQL
    assert d.domain == "structured"


def test_relationship_query_routes_to_graph():
    d = _router().route("what entities are connected to FAISS")
    assert d.retriever is RetrieverTarget.GRAPH


def test_live_query_routes_to_api():
    d = _router().route("what is the current price of an A100 GPU right now")
    assert d.retriever is RetrieverTarget.API


def test_plain_query_routes_to_default_hybrid():
    d = _router().route("what is reciprocal rank fusion")
    assert d.retriever is RetrieverTarget.HYBRID
    assert d.domain == "docs"
    assert d.strategy == "heuristic"


def test_route_never_raises_on_empty():
    d = _router().route("")
    assert d.retriever is RetrieverTarget.HYBRID
