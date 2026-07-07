from src.model.intent_training import build_examples_for_document


def test_build_examples_emit_only_route_labels():
    doc = {"id": "d1", "title": "FAISS", "contents": "vector index library"}
    examples = build_examples_for_document(doc, ["vector", "index", "ranking"])
    labels = {e["label"] for e in examples}
    assert labels <= {"chat", "search", "tool"}
    assert labels == {"chat", "search", "tool"}  # all three represented
