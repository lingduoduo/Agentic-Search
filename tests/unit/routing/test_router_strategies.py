from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.route import RetrieverTarget
from src.internal.routing.router import Router


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_logical_route_picks_named_route():
    router = Router(
        RouteRegistry(DEFAULT_ROUTES), llm=_StubLLM("structured"), logical=True
    )
    d = router.route("break down the dataset somehow")
    assert d.domain == "structured"
    assert d.retriever is RetrieverTarget.SQL
    assert d.strategy == "logical"


def test_logical_route_falls_back_to_heuristic_on_bad_label():
    router = Router(
        RouteRegistry(DEFAULT_ROUTES), llm=_StubLLM("not-a-route"), logical=True
    )
    d = router.route("what is faiss")
    assert d.strategy == "heuristic"  # unknown label → heuristic
    assert d.domain == "docs"


def test_semantic_route_picks_nearest_description():
    # Embedder returns vectors so the "graph" description is nearest to the query.
    def embedder(texts):
        # Map any text containing "connect" near the graph route vector.
        out = []
        for t in texts:
            tl = t.lower()
            if "connect" in tl or "relationship" in tl or "graph" in tl:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    router = Router(RouteRegistry(DEFAULT_ROUTES), embedder=embedder, semantic=True)
    d = router.route("how are these things connected")
    assert d.retriever is RetrieverTarget.GRAPH
    assert d.strategy == "semantic"
