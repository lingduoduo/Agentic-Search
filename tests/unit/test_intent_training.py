import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model import intent_training
from src.model.intent_training import (
    IntentTrainingConfig,
    build_examples_for_document,
    run_intent_training,
)
from src.model.intent_evaluation import (
    IntentPredictionRecord,
    PromotionCriteria,
    compare_for_promotion,
    compose_candidate_cascade,
    evaluate_intent_predictions,
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
    artifact_names = (
        "intent_model.pt",
        "split_manifest.json",
        "evaluation_report.json",
    )
    old_contents = {name: f"previous:{name}".encode() for name in artifact_names}
    for name, contents in old_contents.items():
        (tmp_path / name).write_bytes(contents)
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

    assert {
        name: (tmp_path / name).read_bytes() for name in artifact_names
    } == old_contents
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.rollback"))


def test_training_workflow_restores_complete_generation_when_publication_fails(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    artifact_names = {
        "intent_model.pt",
        "split_manifest.json",
        "evaluation_report.json",
    }
    old_contents = {name: f"previous:{name}".encode() for name in artifact_names}
    for name, contents in old_contents.items():
        (tmp_path / name).write_bytes(contents)

    real_replace = Path.replace
    publications = 0

    def fail_second_publication(path, target):
        nonlocal publications
        target = Path(target)
        if path.suffix == ".tmp" and target.name in artifact_names:
            publications += 1
            if publications == 2:
                raise OSError("injected publication failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_publication)

    with pytest.raises(OSError, match="injected publication failure"):
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

    assert publications == 2
    assert {
        name: (tmp_path / name).read_bytes() for name in artifact_names
    } == old_contents
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.rollback"))


def test_training_workflow_cleans_backup_when_backup_copy_fails(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    artifact_names = (
        "intent_model.pt",
        "split_manifest.json",
        "evaluation_report.json",
    )
    old_contents = {name: f"previous:{name}".encode() for name in artifact_names}
    for name, contents in old_contents.items():
        (tmp_path / name).write_bytes(contents)

    def fail_backup_copy(_source, _target):
        raise OSError("injected backup copy failure")

    monkeypatch.setattr(intent_training.shutil, "copyfile", fail_backup_copy)

    with pytest.raises(OSError, match="injected backup copy failure"):
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

    assert {
        name: (tmp_path / name).read_bytes() for name in artifact_names
    } == old_contents
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.rollback"))


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


@pytest.mark.parametrize(
    "records, diagnostic",
    [
        (
            [
                {
                    "id": None,
                    "title": "Indexing",
                    "contents": "Search docs",
                }
            ],
            "invalid id",
        ),
        (
            [{"id": "doc", "title": None, "contents": "Search docs"}],
            "invalid title",
        ),
        (
            [{"id": "doc", "title": "Indexing", "contents": None}],
            "invalid contents",
        ),
        (
            [
                {"id": "same", "title": "Index A", "contents": "First"},
                {"id": "same", "title": "Index B", "contents": "Second"},
            ],
            "duplicate document identity",
        ),
    ],
)
def test_generate_cli_rejects_nullable_or_duplicate_document_ids(
    tmp_path, capsys, records, diagnostic
):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output_path = tmp_path / "intent_examples.json"

    exit_code = intent_training.main(
        ["generate", "--corpus", str(corpus_path), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert diagnostic in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not output_path.exists()


def test_generate_cli_derives_unique_stable_ids_for_missing_document_ids(
    tmp_path,
):
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        {"title": "Indexing", "contents": "First search document"},
        {"title": "Indexing", "contents": "Second search document"},
    ]
    corpus_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output_path = tmp_path / "intent_examples.json"

    assert (
        intent_training.main(
            ["generate", "--corpus", str(corpus_path), "--output", str(output_path)]
        )
        == 0
    )
    rows = json.loads(output_path.read_text())

    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["source_doc_id"].startswith("sha256:") for row in rows)


def test_generate_cli_validates_generated_rows_before_writing(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        intent_training,
        "generate_intent_examples",
        lambda **_kwargs: [{"text": "missing identity", "label": "chat"}],
    )
    output_path = tmp_path / "intent_examples.json"

    exit_code = intent_training.main(
        [
            "generate",
            "--corpus",
            str(tmp_path / "ignored.jsonl"),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "generated intent example" in captured.err.lower()
    assert not output_path.exists()


def test_training_rejects_baseline_ids_that_do_not_match_test_split(tmp_path):
    baseline = json.loads((FIXTURES / "baseline_predictions.json").read_text())
    baseline.pop()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match test IDs"):
        run_intent_training(
            IntentTrainingConfig(
                examples_path=FIXTURES / "intent_examples.json",
                baseline_path=baseline_path,
                output_dir=tmp_path / "artifacts",
                epochs=1,
                embedding_dim=8,
                hidden_dim=16,
                seed=17,
            )
        )

    assert not (tmp_path / "artifacts").exists()


def test_baseline_cli_generates_regex_records_and_requires_only_ambiguous_captures(
    tmp_path,
):
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(
        json.dumps(
            [
                {
                    "example_id": "chat-b",
                    "expected": "chat",
                    "predicted": "chat",
                    "confidence": 1.0,
                    "latency_ms": 40.0,
                    "mechanism": "classifier",
                },
                {
                    "example_id": "search-c",
                    "expected": "search",
                    "predicted": "search",
                    "confidence": 1.0,
                    "latency_ms": 45.0,
                    "mechanism": "classifier",
                },
                {
                    "example_id": "tool-a",
                    "expected": "tool",
                    "predicted": "tool",
                    "confidence": 1.0,
                    "latency_ms": 50.0,
                    "mechanism": "rule_based",
                },
            ]
        )
    )
    output = tmp_path / "baseline.json"

    exit_code = intent_training.main(
        [
            "baseline",
            "--examples",
            str(FIXTURES / "intent_examples.json"),
            "--fallback-predictions",
            str(fallback_path),
            "--output",
            str(output),
            "--seed",
            "17",
        ]
    )

    assert exit_code == 0
    records = json.loads(output.read_text())
    assert {record["example_id"] for record in records} == {
        "chat-b",
        "chat-b-regex",
        "search-c",
        "search-c-regex",
        "tool-a",
        "tool-a-regex",
    }
    tool = next(record for record in records if record["example_id"] == "tool-a-regex")
    assert tool["mechanism"] == "regex"
    assert tool["predicted"] == "tool"


def test_baseline_cli_rejects_missing_ambiguous_capture(tmp_path, capsys):
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text("[]")

    exit_code = intent_training.main(
        [
            "baseline",
            "--examples",
            str(FIXTURES / "intent_examples.json"),
            "--fallback-predictions",
            str(fallback_path),
            "--output",
            str(tmp_path / "baseline.json"),
            "--seed",
            "17",
        ]
    )

    assert exit_code == 1
    assert "ambiguous held-out" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["train", "--examples"],
        ["unknown-command"],
    ],
)
def test_cli_usage_errors_return_one_without_artifacts(argv, tmp_path, capsys):
    exit_code = intent_training.main(argv)

    assert exit_code == 1
    assert "usage:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_train_cli_returns_zero_for_promotable_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        intent_training,
        "run_intent_training",
        lambda _config: SimpleNamespace(promotion=SimpleNamespace(promotable=True)),
    )

    exit_code = intent_training.main(
        [
            "train",
            "--examples",
            str(tmp_path / "examples.json"),
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert exit_code == 0


def test_train_cli_returns_two_for_real_nonpromotable_workflow(tmp_path):
    exit_code = intent_training.main(
        [
            "train",
            "--examples",
            str(FIXTURES / "intent_examples.json"),
            "--baseline",
            str(FIXTURES / "baseline_predictions.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--epochs",
            "1",
            "--embedding-dim",
            "8",
            "--hidden-dim",
            "16",
            "--seed",
            "17",
        ]
    )

    assert exit_code == 2
    assert (tmp_path / "artifacts" / "intent_model.pt").exists()


def test_deterministic_fixture_candidate_passes_default_promotion_gates():
    baseline_payload = json.loads((FIXTURES / "baseline_predictions.json").read_text())
    candidate_payload = json.loads(
        (FIXTURES / "promotable_candidate_predictions.json").read_text()
    )
    baseline = tuple(IntentPredictionRecord(**record) for record in baseline_payload)
    model = tuple(IntentPredictionRecord(**record) for record in candidate_payload)
    threshold = 0.95

    candidate = compose_candidate_cascade(model, baseline, threshold=threshold)
    decision = compare_for_promotion(
        evaluate_intent_predictions(candidate, threshold=threshold),
        evaluate_intent_predictions(baseline, threshold=threshold),
        PromotionCriteria(),
    )

    assert decision.promotable is True
    assert all(gate["passed"] for gate in decision.gates)
    assert all(
        candidate_record == baseline_record
        for candidate_record, baseline_record in zip(candidate, baseline)
        if baseline_record.mechanism == "regex"
    )


def test_train_cli_returns_one_for_malformed_input_without_traceback(tmp_path, capsys):
    exit_code = intent_training.main(
        [
            "train",
            "--examples",
            str(tmp_path / "missing-examples.json"),
            "--baseline",
            str(tmp_path / "missing-baseline.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "intent training failed:" in captured.err
    assert "Traceback" not in captured.err
