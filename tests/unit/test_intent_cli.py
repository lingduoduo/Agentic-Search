import json
from pathlib import Path

import numpy as np

from src.model.pre_training.intents import cli as intent_index_cli
from src.model.pre_training.intents import data as intent_index_data
from src.model.pre_training.intents import evaluation as intent_index_eval
from src.model.pre_training.intents.model import INDEX_FILENAME, IntentIndex

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _fake_encode(texts, *, model_name="test-encoder"):
    """Encode by the route word each text starts with, onto a basis axis."""
    rows = []
    for text in texts:
        axis = _AXIS.get(text.split()[0], 0)
        rows.append(np.eye(3, dtype=np.float32)[axis])
    return np.stack(rows)


def _canonical(tmp_path: Path, count: int = 3) -> Path:
    records = [
        {
            "id": f"{route}-{position}",
            "text": f"{route} canonical text {position}",
            "route": route,
            "modules": [_MODULE[route]],
        }
        for route in _AXIS
        for position in range(count)
    ]
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_build_writes_a_loadable_index(tmp_path):
    output = tmp_path / "index"

    index = intent_index_data.build_index(
        _canonical(tmp_path), output, model_name="test-encoder", encode=_fake_encode
    )

    assert index.size == 9
    reloaded = IntentIndex.load(output / INDEX_FILENAME)
    assert reloaded.size == 9
    assert reloaded.encoder == "test-encoder"


def test_build_fingerprints_the_canonical_file(tmp_path):
    """A stale index against an edited canonical file must be detectable."""
    canonical = _canonical(tmp_path)
    first = intent_index_data.build_index(
        canonical, tmp_path / "a", model_name="test-encoder", encode=_fake_encode
    )

    records = json.loads(canonical.read_text(encoding="utf-8"))
    records[0]["text"] = "search canonical text edited"
    canonical.write_text(json.dumps(records), encoding="utf-8")
    second = intent_index_data.build_index(
        canonical, tmp_path / "b", model_name="test-encoder", encode=_fake_encode
    )

    assert first.fingerprint != second.fingerprint


def test_leakage_check_flags_an_eval_query_identical_to_a_canonical_example(tmp_path):
    index = intent_index_data.build_index(
        _canonical(tmp_path),
        tmp_path / "index",
        model_name="test-encoder",
        encode=_fake_encode,
    )
    texts = ["search canonical text 0"]

    leaks = intent_index_eval.check_leakage(index, texts, _fake_encode(texts))

    assert leaks and "search canonical text 0" in leaks[0]


def test_leakage_check_flags_a_near_duplicate_above_the_cosine_bar(tmp_path):
    """With kNN the index IS the model, so overlap manufactures accuracy."""
    index = intent_index_data.build_index(
        _canonical(tmp_path),
        tmp_path / "index",
        model_name="test-encoder",
        encode=_fake_encode,
    )
    texts = ["search something else entirely"]

    leaks = intent_index_eval.check_leakage(index, texts, _fake_encode(texts))

    assert leaks


def test_leakage_check_passes_for_a_genuinely_distinct_query(tmp_path):
    index = intent_index_data.build_index(
        _canonical(tmp_path),
        tmp_path / "index",
        model_name="test-encoder",
        encode=_fake_encode,
    )
    texts = ["unrelated"]
    vectors = np.array([[0.0, 0.6, 0.8]], dtype=np.float32)

    assert intent_index_eval.check_leakage(index, texts, vectors) == []


def test_build_command_reports_low_support_modules(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(intent_index_data, "encode_texts", _fake_encode)
    output = tmp_path / "index"

    exit_code = intent_index_cli.main(
        ["build", "--canonical", str(_canonical(tmp_path)), "--output", str(output)]
    )

    assert exit_code == 0
    assert "low support" in capsys.readouterr().out.lower()


def test_build_command_reports_a_missing_canonical_file(tmp_path, capsys):
    exit_code = intent_index_cli.main(
        [
            "build",
            "--canonical",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "index"),
        ]
    )

    assert exit_code == 1
    assert "missing.json" in capsys.readouterr().err
