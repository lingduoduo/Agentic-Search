import json

from src.internal.retrieval.eval_metrics import routing_accuracy
from src.internal.retrieval.eval_runner import run_routing_eval


def test_routing_accuracy_basic():
    assert routing_accuracy(
        ["sql", "hybrid", "graph"], ["sql", "hybrid", "api"]
    ) == round(2 / 3, 4)
    assert routing_accuracy([], []) == 0.0


def test_run_routing_eval_on_labeled_set(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        {"query": "how many papers per year", "retriever": "sql"},
        {"query": "what is reciprocal rank fusion", "retriever": "hybrid"},
        {"query": "what is connected to FAISS", "retriever": "graph"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = run_routing_eval(str(path))
    assert out["num_queries"] == 3
    assert out["routing_accuracy"] == 1.0


def test_repo_routing_labels_meet_threshold():
    out = run_routing_eval("data/eval/routing_labels.jsonl")
    assert out["routing_accuracy"] >= 0.8  # committed gate threshold
