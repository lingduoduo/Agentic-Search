from src.internal.routing.routing_factory import build_router_from_env
from src.internal.routing.route import RetrieverTarget


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ROUTING_ENABLED", raising=False)
    assert build_router_from_env() is None


def test_enabled_builds_heuristic_router(monkeypatch):
    monkeypatch.setenv("ROUTING_ENABLED", "1")
    monkeypatch.delenv("ROUTING_LOGICAL", raising=False)
    monkeypatch.delenv("ROUTING_SEMANTIC", raising=False)
    router = build_router_from_env()
    assert router is not None
    d = router.route("how many documents are indexed")
    assert d.retriever is RetrieverTarget.SQL
