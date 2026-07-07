import json

import pytest

from src.model.intent_classifier import IntentPipeline
from src.model.intent_distillation import (
    build_distillation_examples,
    distill_and_train,
    label_query,
    load_queries_from_file,
)


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, _messages, **_kwargs):
        self.calls += 1
        return self.reply


def test_label_query_regex_path_ignores_llm():
    llm = _FakeLLM("tool")
    label, teacher = label_query("find the Q3 report", llm=llm)
    assert (label, teacher) == ("search", "regex")
    assert llm.calls == 0


def test_label_query_llm_tail_when_regex_defers():
    # A query the anchored regex returns None for, with currency cue deferral.
    label, teacher = label_query(
        "the current bitcoin price trend", llm=_FakeLLM("search")
    )
    assert teacher == "llm"
    assert label in {"chat", "search", "tool"}


def test_label_query_rule_based_when_no_llm():
    label, teacher = label_query("the current bitcoin price trend", llm=None)
    assert teacher == "rule_based"


def test_label_query_llm_error_falls_to_rule_based():
    class _Boom:
        def complete(self, *_a, **_k):
            raise RuntimeError("boom")

    label, teacher = label_query("the current bitcoin price trend", llm=_Boom())
    assert teacher == "rule_based"


def test_build_examples_shape_and_drops_blanks():
    ex = build_distillation_examples(["find X", "  ", "what is Y"])
    assert ex == [
        {"text": "find X", "label": "search"},
        {"text": "what is Y", "label": "chat"},
    ]


def test_distill_and_train_roundtrip(tmp_path):
    queries = (
        ["find " + t for t in ("faiss", "bm25", "hnsw")]
        + ["what is " + t for t in ("faiss", "bm25", "hnsw")]
        + ["create a ticket for " + t for t in ("faiss", "bm25", "hnsw")]
    )
    pt = tmp_path / "m.pt"
    ex = tmp_path / "ex.json"
    result = distill_and_train(
        queries, output_path=pt, examples_path=ex, epochs=40, min_freq=1
    )
    assert result.num_examples == len(queries)
    assert sum(result.teacher_counts.values()) == len(queries)
    assert set(result.label_counts) <= {"chat", "search", "tool"}
    reloaded = IntentPipeline.load(str(pt))
    assert reloaded.predict_text("find hnsw").intent in {"chat", "search", "tool"}


def test_distill_and_train_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        distill_and_train(
            [], output_path=tmp_path / "m.pt", examples_path=tmp_path / "e.json"
        )


def test_load_queries_from_file_txt_and_json(tmp_path):
    txt = tmp_path / "q.txt"
    txt.write_text("find X\n\nwhat is Y\n")
    assert load_queries_from_file(txt) == ["find X", "what is Y"]
    js = tmp_path / "q.json"
    js.write_text(json.dumps(["a", {"text": "b"}, {"question": "c"}]))
    assert load_queries_from_file(js) == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        load_queries_from_file(tmp_path / "missing.txt")
