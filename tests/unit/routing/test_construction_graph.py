from src.internal.routing.construction.graph import (
    KnowledgeGraphQueryConstructor,
    validate_cypher,
)
from src.internal.routing.route import RetrieverTarget, RouteDecision


def _route():
    return RouteDecision(
        domain="graph",
        sources=["knowledge_graph"],
        retriever=RetrieverTarget.GRAPH,
        construction_target=RetrieverTarget.GRAPH,
    )


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages, **kwargs):
        return self._reply


def test_validate_accepts_match_return():
    assert validate_cypher('MATCH (n {name: "FAISS"})-[r]-(m) RETURN n, r, m')


def test_validate_rejects_writes():
    assert not validate_cypher('CREATE (n:Node {name: "x"}) RETURN n')
    assert not validate_cypher("MATCH (n) DETACH DELETE n")


def test_constructor_builds_cypher_from_entity():
    out = KnowledgeGraphQueryConstructor(
        _StubLLM('{"entity": "FAISS", "relation": "uses"}')
    ).construct("what is connected to FAISS", _route())
    assert out.target is RetrieverTarget.GRAPH
    assert out.payload["entity"] == "FAISS"
    assert "MATCH" in out.payload["cypher"]
    assert validate_cypher(out.payload["cypher"])


def test_constructor_degrades_on_bad_json():
    out = KnowledgeGraphQueryConstructor(_StubLLM("not json")).construct("x", _route())
    assert out.payload["cypher"] is None
    assert out.payload["entity"] is None


def test_validate_rejects_set_without_trailing_space():
    assert not validate_cypher("MATCH (n) SET\nn.x = 1 RETURN n")


def test_entity_with_write_keyword_substring_still_builds():
    out = KnowledgeGraphQueryConstructor(
        _StubLLM('{"entity": "Drop Table Co", "relation": "uses"}')
    ).construct("what connects to Drop Table Co", _route())
    assert out.payload["entity"] == "Drop Table Co"
    assert out.payload["cypher"] is not None
    assert "Drop Table Co" in out.payload["cypher"]
