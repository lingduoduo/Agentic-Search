import json
from pathlib import Path

import pytest

from src.model.intent_training import (
    IntentTrainingConfig,
    build_examples_for_document,
    run_intent_training,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "intent"


def test_build_examples_emit_only_route_labels():
    doc = {"id": "d1", "title": "FAISS", "contents": "vector index library"}
    examples = build_examples_for_document(doc, ["vector", "index", "ranking"])
    labels = {e["label"] for e in examples}
    assert labels <= {"chat", "search", "tool"}
    assert labels == {"chat", "search", "tool"}  # all three represented


def test_training_workflow_writes_artifact_manifest_and_report(tmp_path):
    pytest.importorskip("torch")

    run = run_intent_training(
        IntentTrainingConfig(
            examples_path=FIXTURES / "intent_examples.json",
            baseline_path=FIXTURES / "baseline_predictions.json",
            output_dir=tmp_path,
            epochs=1,
            embedding_dim=8,
            hidden_dim=16,
            seed=17,
        )
    )

    assert run.checkpoint_path.exists()
    assert (tmp_path / "split_manifest.json").exists()
    report = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert report["labels"] == ["chat", "search", "tool"]
    assert "promotion" in report
