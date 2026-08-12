import json
from pathlib import Path

import pytest

from src.model import intent_training
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


def test_training_workflow_publishes_nothing_when_artifact_staging_fails(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    real_mkstemp = intent_training.tempfile.mkstemp
    calls = 0

    def fail_second_staging_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(intent_training.tempfile, "mkstemp", fail_second_staging_call)

    with pytest.raises(OSError, match="injected staging failure"):
        run_intent_training(
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

    assert not (tmp_path / "intent_model.pt").exists()
    assert not (tmp_path / "split_manifest.json").exists()
    assert not (tmp_path / "evaluation_report.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_generate_cli_reports_malformed_corpus_without_traceback(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("[1, 2, 3]\n", encoding="utf-8")

    exit_code = intent_training.main(
        [
            "generate",
            "--corpus",
            str(corpus_path),
            "--output",
            str(tmp_path / "intent_examples.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "intent training failed:" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "intent_examples.json").exists()


def test_generate_cli_rejects_empty_corpus(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("", encoding="utf-8")

    exit_code = intent_training.main(
        [
            "generate",
            "--corpus",
            str(corpus_path),
            "--output",
            str(tmp_path / "intent_examples.json"),
        ]
    )

    assert exit_code == 1
    assert "contains no documents" in capsys.readouterr().err
    assert not (tmp_path / "intent_examples.json").exists()


def test_generate_cli_reports_malformed_vocabulary_without_traceback(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id": "doc", "title": "Indexing", "contents": "Search docs"}\n',
        encoding="utf-8",
    )
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text("[]", encoding="utf-8")

    exit_code = intent_training.main(
        [
            "generate",
            "--corpus",
            str(corpus_path),
            "--vocabulary",
            str(vocabulary_path),
            "--output",
            str(tmp_path / "intent_examples.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "intent training failed:" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "intent_examples.json").exists()
