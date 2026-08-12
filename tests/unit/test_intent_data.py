from pathlib import Path

import pytest

from src.model.intent_data import (
    IntentEvalQuery,
    load_intent_eval_queries,
    load_intent_examples,
    split_intent_examples,
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


def test_grouped_split_is_reproducible_and_has_no_source_leakage(tmp_path: Path):
    rows = []
    for label in ("chat", "search", "tool"):
        for group in range(10):
            rows.append(
                {
                    "id": f"{label}-{group}-a",
                    "text": f"{label} {group} a",
                    "label": label,
                    "source": f"{label}-{group}",
                }
            )
            rows.append(
                {
                    "id": f"{label}-{group}-b",
                    "text": f"{label} {group} b",
                    "label": label,
                    "source": f"{label}-{group}",
                }
            )
    path = tmp_path / "examples.json"
    path.write_text(__import__("json").dumps(rows))
    examples = load_intent_examples(path)
    first = split_intent_examples(examples, seed=23)
    second = split_intent_examples(examples, seed=23)
    assert first == second
    sources = [
        {item.source for item in first.train},
        {item.source for item in first.validation},
        {item.source for item in first.test},
    ]
    assert not (
        sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2]
    )


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
