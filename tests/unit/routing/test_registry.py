import json

from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.route import RetrieverTarget


def test_default_registry_has_a_default_hybrid_route():
    reg = RouteRegistry(DEFAULT_ROUTES)
    assert reg.default().retriever is RetrieverTarget.HYBRID
    assert reg.get("docs") is not None


def test_by_retriever_lookup():
    reg = RouteRegistry(DEFAULT_ROUTES)
    assert reg.by_retriever(RetrieverTarget.SQL).retriever is RetrieverTarget.SQL
    assert reg.by_retriever(RetrieverTarget.METADATA) is None  # not a registered route


def test_from_file_loads_custom_routes(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "wiki",
                    "description": "internal wiki",
                    "sources": ["wiki"],
                    "retriever": "dense",
                }
            ]
        )
    )
    reg = RouteRegistry.from_file(str(path))
    assert reg.default().name == "wiki"
    assert reg.default().retriever is RetrieverTarget.DENSE


def test_from_env_without_path_uses_defaults(monkeypatch):
    monkeypatch.delenv("ROUTING_REGISTRY_PATH", raising=False)
    reg = RouteRegistry.from_env()
    assert reg.get("docs") is not None
