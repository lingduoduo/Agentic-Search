import json
from pathlib import Path

import pytest

from src.model.intent_data import (
    IntentEvalQuery,
    load_canonical_examples,
    load_intent_eval_queries,
    load_intent_examples,
)


def test_load_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "examples.json"
    path.write_text('[{"id":"x","text":"buy it","label":"purchase","source":"manual"}]')
    with pytest.raises(ValueError, match="purchase"):
        load_intent_examples(path)


def test_load_rejects_conflicting_duplicate_text(tmp_path: Path):
    path = tmp_path / "examples.json"
    path.write_text(
        '[{"id":"a","text":"open the report","label":"search","source":"s1"},'
        '{"id":"b","text":"open the report","label":"tool","source":"s2"}]'
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_intent_examples(path)


def test_load_out_of_scope_probes_rejects_labels_and_duplicates(tmp_path):
    from src.model.intent_data import load_out_of_scope_probes

    path = tmp_path / "oos.json"
    path.write_text('[{"id": "a", "text": "my cat knocked over the plant"}]')
    assert load_out_of_scope_probes(path) == (("a", "my cat knocked over the plant"),)

    path.write_text('[{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]')
    with pytest.raises(ValueError, match="Duplicate"):
        load_out_of_scope_probes(path)

    path.write_text('[{"id": "a", "text": ""}]')
    with pytest.raises(ValueError, match="text"):
        load_out_of_scope_probes(path)


def test_load_intent_eval_queries_reads_id_text_and_label(tmp_path: Path):
    path = tmp_path / "eval.json"
    path.write_text(
        '[{"id": "e1", "text": "where did we land on the index rebuild",'
        ' "label": "search"}]',
        encoding="utf-8",
    )

    queries = load_intent_eval_queries(path)

    assert queries == (
        IntentEvalQuery(
            id="e1", text="where did we land on the index rebuild", label="search"
        ),
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ('[{"id": "e1", "text": "hi", "label": "purchase"}]', "Unknown intent label"),
        ('[{"id": "e1", "text": " ", "label": "chat"}]', "empty 'text'"),
        (
            '[{"id": "e1", "text": "a", "label": "chat"},'
            ' {"id": "e1", "text": "b", "label": "chat"}]',
            "Duplicate",
        ),
        ("[]", "no records"),
    ],
)
def test_load_intent_eval_queries_rejects_invalid_records(
    tmp_path: Path, payload: str, message: str
):
    path = tmp_path / "eval.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_intent_eval_queries(path)


def _canonical_records():
    return [
        {
            "id": "canon-001",
            "text": "what is the current price of bitcoin",
            "route": "search",
            "modules": ["current_info", "lookup_fact"],
        },
        {
            "id": "canon-002",
            "text": "summarize the Q3 earnings report",
            "route": "chat",
            "modules": ["summarize"],
        },
    ]


def _write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_examples_round_trip_with_multiple_modules(tmp_path):
    path = _write_json(tmp_path, "canonical.json", _canonical_records())

    examples = load_canonical_examples(path)

    assert len(examples) == 2
    assert examples[0].route == "search"
    assert examples[0].modules == ("current_info", "lookup_fact")
    assert examples[1].modules == ("summarize",)


def test_canonical_rejects_a_module_from_another_route(tmp_path):
    records = _canonical_records()
    records[0]["modules"] = ["current_info", "summarize"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="summarize"):
        load_canonical_examples(path)


def test_canonical_rejects_an_empty_module_list(tmp_path):
    records = _canonical_records()
    records[0]["modules"] = []
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="at least one"):
        load_canonical_examples(path)


def test_canonical_rejects_a_missing_modules_field(tmp_path):
    records = _canonical_records()
    del records[0]["modules"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="modules"):
        load_canonical_examples(path)


def test_canonical_rejects_an_unknown_route(tmp_path):
    records = _canonical_records()
    records[0]["route"] = "browse"
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="browse"):
        load_canonical_examples(path)


def test_canonical_rejects_duplicate_ids(tmp_path):
    records = _canonical_records()
    records[1]["id"] = records[0]["id"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="Duplicate"):
        load_canonical_examples(path)


def test_canonical_rejects_duplicate_text(tmp_path):
    """Duplicated text silently doubles a point's pull on every query."""
    records = _canonical_records()
    records[1]["text"] = records[0]["text"].upper()
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_canonical_examples(path)


def test_canonical_rejects_an_empty_file(tmp_path):
    path = _write_json(tmp_path, "canonical.json", [])

    with pytest.raises(ValueError, match="no records"):
        load_canonical_examples(path)


def test_eval_queries_carry_modules_when_present(tmp_path):
    path = _write_json(
        tmp_path,
        "eval.json",
        [
            {
                "id": "eval-001",
                "text": "find the Q3 earnings report",
                "label": "search",
                "modules": ["lookup_document"],
            }
        ],
    )

    queries = load_intent_eval_queries(path)

    assert queries[0].modules == ("lookup_document",)


def test_eval_queries_without_modules_still_load(tmp_path):
    """The legacy 30 queries predate modules; loading must not break."""
    path = _write_json(
        tmp_path,
        "eval.json",
        [{"id": "eval-001", "text": "find the report", "label": "search"}],
    )

    assert load_intent_eval_queries(path)[0].modules == ()


def test_eval_queries_reject_a_module_from_another_route(tmp_path):
    path = _write_json(
        tmp_path,
        "eval.json",
        [
            {
                "id": "eval-001",
                "text": "find the report",
                "label": "search",
                "modules": ["summarize"],
            }
        ],
    )

    with pytest.raises(ValueError, match="summarize"):
        load_intent_eval_queries(path)
