from pathlib import Path

import pytest

from examples.prepare_local_rag_smoke_dataset import build_smoke_records


REPO_CORPUS = Path(__file__).parents[2] / "data" / "corpus.jsonl"


def test_build_smoke_records_retrieves_expected_context():
    records = build_smoke_records(REPO_CORPUS, topk=1)

    assert records
    first = records[0]
    assert set(first) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert "FAISS" in first["prompt"][0]["content"]
    assert first["reward_model"]["ground_truth"]["target"] == ["FAISS"]
    assert first["extra_info"] == {"split": "smoke", "index": 0}


def test_build_smoke_records_rejects_non_positive_topk():
    with pytest.raises(ValueError, match="topk must be at least 1"):
        build_smoke_records(REPO_CORPUS, topk=0)


def test_build_smoke_records_rejects_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        build_smoke_records(tmp_path / "missing.jsonl")


def test_build_smoke_records_rejects_empty_corpus(tmp_path):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Corpus file is empty"):
        build_smoke_records(corpus)


def test_build_smoke_records_names_question_when_retrieval_is_empty(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"1","title":"Unrelated","contents":"xyzzy plugh"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No retrieval results for smoke question"):
        build_smoke_records(corpus)


def test_build_smoke_records_surfaces_malformed_jsonl(tmp_path):
    corpus = tmp_path / "broken.jsonl"
    corpus.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed corpus JSONL"):
        build_smoke_records(corpus)
