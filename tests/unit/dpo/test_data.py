"""Tests for the DPO preference-pair loader.

Validation is strict on purpose: these pairs *are* the training signal, so a
silently skipped row changes what the model learns with no later symptom.
"""

from __future__ import annotations

import json

import pytest

from src.model.post_training.dpo.data import PreferenceExample, load_preference_pairs


def _write(tmp_path, lines):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(prompt="What is FAISS?", chosen="A vector search library.", rejected="A DB."):
    return json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected})


def test_loads_well_formed_pairs(tmp_path):
    path = _write(tmp_path, [_row(), _row(prompt="What is BM25?")])

    pairs = load_preference_pairs(path)

    assert len(pairs) == 2
    assert pairs[0] == PreferenceExample(
        prompt="What is FAISS?",
        chosen="A vector search library.",
        rejected="A DB.",
    )
    assert pairs[1].prompt == "What is BM25?"


def test_blank_lines_are_skipped_not_rejected(tmp_path):
    """Trailing and interleaved blank lines are formatting, not corruption."""
    path = _write(tmp_path, [_row(), "", "   ", _row(prompt="Second?")])

    assert len(load_preference_pairs(path)) == 2


def test_malformed_json_names_the_line(tmp_path):
    path = _write(tmp_path, [_row(), "{not json", _row()])

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    # A loader that says only "invalid file" is useless against a large dataset.
    assert "line 2" in str(exc.value)


def test_non_object_line_is_rejected(tmp_path):
    path = _write(tmp_path, [json.dumps(["prompt", "chosen", "rejected"])])

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    assert "line 1" in str(exc.value)


@pytest.mark.parametrize("missing", ["prompt", "chosen", "rejected"])
def test_missing_key_is_rejected(tmp_path, missing):
    record = json.loads(_row())
    del record[missing]
    path = _write(tmp_path, [json.dumps(record)])

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    assert missing in str(exc.value)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_value_is_rejected(tmp_path, blank):
    path = _write(tmp_path, [_row(chosen=blank)])

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    assert "chosen" in str(exc.value)


def test_identical_chosen_and_rejected_is_rejected(tmp_path):
    """Such a pair contributes constant log 2 loss and zero gradient forever.

    It is never a useful training example, and it is a common artifact of a
    pair-construction bug upstream — so it is worth failing loudly on.
    """
    path = _write(tmp_path, [_row(chosen="same text", rejected="same text")])

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    assert "identical" in str(exc.value).lower()


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_preference_pairs(path)

    assert "no preference pairs" in str(exc.value).lower()


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preference_pairs(tmp_path / "nope.jsonl")
