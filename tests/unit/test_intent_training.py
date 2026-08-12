import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.internal.document_index.text import tokenize_text
from src.model import intent_training
from src.model.intent_data import load_intent_examples, split_intent_examples
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


def test_build_examples_emit_ambiguous_cases_for_every_label():
    """The learned route only matters where the regex router defers.

    A corpus of phrases the deterministic router already decides leaves the
    model with no held-out coverage, so generation must emit requests that
    regex defers for each label.
    """
    from src.internal.servers.web.intent_routing import _regex_route

    doc = {"id": "d1", "title": "FAISS", "contents": "vector index library"}
    examples = build_examples_for_document(doc, ["vector", "index", "ranking"])
    deferred = [e for e in examples if _regex_route(e["text"]) is None]

    assert {e["label"] for e in deferred} == {"chat", "search", "tool"}
    assert all({"ambiguous", "multi_intent"} & set(e["tags"]) for e in deferred)
    assert len(deferred) * 3 >= len(examples)  # at least a third stay ambiguous


def test_build_examples_label_multi_intent_requests_by_route_precedence():
    """A request carrying two intents takes the higher-precedence route.

    The dispatcher resolves tool > search > chat, so a request that both looks
    something up and takes an action must train as ``tool``.
    """
    from src.internal.servers.web.intent_routing import _regex_route

    doc = {"id": "d1", "title": "FAISS", "contents": "vector index library"}
    multi = [
        e
        for e in build_examples_for_document(doc, ["vector", "index", "ranking"])
        if "multi_intent" in e["tags"]
    ]

    assert {e["label"] for e in multi} == {"search", "tool"}
    assert all(_regex_route(e["text"]) is None for e in multi)


def test_neutral_fillers_appear_equally_often_under_every_label():
    """Chatter must carry no class evidence, or out-of-scope text scores high.

    A token's embedding is pulled by the labels it trains under. Balancing the
    neutral slots leaves them class-neutral, so a request built only from
    ordinary English pools toward the centre and its softmax stays near 1/3 —
    the only abstention signal a three-label model has.
    """
    counts = {slot: Counter() for slot in ("role", "time", "artifact", "opener")}
    for _, template, label, _ in intent_training._FRAMES:
        for slot, per_label in counts.items():
            if "{" + slot + "}" in template:
                per_label[label] += 1

    for slot, per_label in counts.items():
        assert set(per_label) == {"chat", "search", "tool"}, slot
        assert len(set(per_label.values())) == 1, (slot, per_label)
        assert min(per_label.values()) >= 2, (slot, per_label)


def test_action_verbs_stay_tool_only():
    """{action} is genuine tool evidence and must not be neutralised."""
    labels = {
        label
        for _, template, label, _ in intent_training._FRAMES
        if "{action}" in template
    }
    assert labels == {"tool"}


def test_frames_introduce_function_words_absent_from_the_corpus():
    """The corpus contributes domain nouns; real queries need ordinary English."""
    document = {"id": "d1", "title": "vector search", "contents": "dense retrieval"}
    corpus_tokens = set(tokenize_text(f"{document['title']} {document['contents']}"))

    generated_tokens: set[str] = set()
    for index in range(4):  # cover every filler rotation
        for example in build_examples_for_document(document, [], document_index=index):
            generated_tokens |= set(tokenize_text(example["text"]))

    for function_word in ("where", "we", "need", "from", "can", "you", "before"):
        assert function_word in generated_tokens
        assert function_word not in corpus_tokens


def test_frames_group_by_frame_and_produce_unique_ids():
    documents = [
        {"id": "d1", "title": "vector search", "contents": "dense retrieval"},
        {"id": "d2", "title": "bm25 ranking", "contents": "sparse retrieval"},
    ]
    examples = [
        example
        for index, document in enumerate(documents)
        for example in build_examples_for_document(document, [], document_index=index)
    ]

    ids = [example["id"] for example in examples]
    assert len(ids) == len(set(ids))
    assert all(example["source"].startswith("frame:") for example in examples)
    by_source: dict[str, set[str]] = {}
    for example in examples:
        by_source.setdefault(example["source"], set()).add(example["id"])
    assert all(len(members) == len(documents) for members in by_source.values())


def test_generated_examples_split_without_source_leakage(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"id": f"d{index}", "title": f"topic {index}", "contents": "x"})
            for index in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    examples_path = tmp_path / "examples.json"
    intent_training.write_intent_examples(
        intent_training.generate_intent_examples(corpus_path=corpus), examples_path
    )

    split = split_intent_examples(load_intent_examples(examples_path), seed=17)

    train_sources = {example.source for example in split.train}
    assert not train_sources & {example.source for example in split.validation}
    assert not train_sources & {example.source for example in split.test}


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


_HAND_WRITTEN_EVAL_TEXT = {
    "chat": "not sure I follow how the two rankings get merged",
    "search": "where did the quarterly numbers file end up",
    "tool": "ping the on-call engineer about the failing job",
}


def _run_fixture_training(
    tmp_path, *, with_eval_queries: bool, copy_training_text: bool = False
):
    eval_queries_path = None
    if with_eval_queries:
        examples = json.loads(
            (FIXTURES / "intent_examples.json").read_text(encoding="utf-8")
        )
        trained_text_by_label: dict[str, str] = {}
        for example in examples:
            trained_text_by_label.setdefault(example["label"], example["text"])
        eval_queries_path = tmp_path / "eval_queries.json"
        eval_queries_path.write_text(
            json.dumps(
                [
                    {
                        "id": f"eval-{label}",
                        "text": (
                            trained_text_by_label[label]
                            if copy_training_text
                            else _HAND_WRITTEN_EVAL_TEXT[label]
                        ),
                        "label": label,
                    }
                    for label in ("chat", "search", "tool")
                ]
            ),
            encoding="utf-8",
        )

    return run_intent_training(
        IntentTrainingConfig(
            examples_path=FIXTURES / "intent_examples.json",
            baseline_path=FIXTURES / "baseline_predictions.json",
            eval_queries_path=eval_queries_path,
            output_dir=tmp_path,
            epochs=1,
            embedding_dim=8,
            hidden_dim=16,
            seed=17,
        )
    )


def test_training_report_records_realistic_accuracy(tmp_path):
    pytest.importorskip("torch")
    run = _run_fixture_training(tmp_path, with_eval_queries=True)

    report = json.loads(run.evaluation_report_path.read_text(encoding="utf-8"))
    assert report["realistic_accuracy"]["total_queries"] == 3
    assert 0.0 <= report["realistic_accuracy"]["accuracy"] <= 1.0
    assert set(report["realistic_accuracy"]) >= {
        "accuracy",
        "coverage",
        "covered_accuracy",
        "macro_f1",
        "per_label_metrics",
        "threshold",
        "total_queries",
    }


def test_training_report_records_null_realistic_accuracy_without_a_set(tmp_path):
    pytest.importorskip("torch")
    run = _run_fixture_training(tmp_path, with_eval_queries=False)

    report = json.loads(run.evaluation_report_path.read_text(encoding="utf-8"))
    assert report["realistic_accuracy"] is None


def test_training_rejects_an_evaluation_set_the_generator_produces(tmp_path):
    """A set the training data already contains cannot measure generalization."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cannot measure generalization"):
        _run_fixture_training(tmp_path, with_eval_queries=True, copy_training_text=True)


def test_promotable_training_checkpoint_stores_selected_threshold(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    real_compare = intent_training.compare_for_promotion

    def force_promotable(*args, **kwargs):
        decision = real_compare(*args, **kwargs)
        return intent_training.PromotionDecision(True, decision.gates, ())

    monkeypatch.setattr(intent_training, "compare_for_promotion", force_promotable)
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

    checkpoint = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["promoted_min_confidence"] == run.selected_threshold


def test_nonpromotable_training_checkpoint_is_inspection_only(tmp_path):
    torch = pytest.importorskip("torch")
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

    checkpoint = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    assert run.promotion.promotable is False
    assert checkpoint["promoted_min_confidence"] is None


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
        evaluate_intent_predictions(
            candidate, threshold=threshold, out_of_scope_abstention=1.0
        ),
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
