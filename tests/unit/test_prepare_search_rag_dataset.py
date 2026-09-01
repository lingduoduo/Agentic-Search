"""Corpus resolution for the QA + retrieval-cache RAG dataset builder."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from examples.prepare_search_rag_dataset import load_corpus_by_id, parse_args


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(
            json.dumps(doc) + "\n"
            for doc in (
                {"id": 1, "title": "FAISS", "contents": "vector search"},
                {"id": 2, "title": "BM25", "contents": "sparse ranking"},
            )
        ),
        encoding="utf-8",
    )
    return path


def test_corpus_path_still_loads_and_keys_by_string_id(corpus_file):
    corpus = load_corpus_by_id(str(corpus_file))

    assert set(corpus) == {"1", "2"}
    assert corpus["1"]["title"] == "FAISS"


def test_registered_corpus_name_resolves_through_the_manifest(corpus_file, monkeypatch):
    """`--corpus scifact` must mean the same corpus here as at the servers."""
    import src.internal.servers.retrieval.corpus_registry as registry

    monkeypatch.setattr(
        registry, "load_manifest", lambda *a, **k: {"demo": {"path": str(corpus_file)}}
    )

    corpus = load_corpus_by_id("demo")

    assert set(corpus) == {"1", "2"}


def test_unknown_corpus_name_is_reported(monkeypatch):
    import src.internal.servers.retrieval.corpus_registry as registry

    monkeypatch.setattr(registry, "load_manifest", lambda *a, **k: {"demo": {}})

    with pytest.raises(ValueError, match="Unknown corpus spec"):
        load_corpus_by_id("nope")


def test_parse_args_takes_a_corpus_name_instead_of_a_path(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_search_rag_dataset",
            "--corpus",
            "scifact",
            "--train_retrieval_cache",
            "train.json",
            "--test_retrieval_cache",
            "test.json",
        ],
    )

    args = parse_args()

    assert args.corpus == "scifact"
    assert args.corpus_path is None


def test_parse_args_requires_one_corpus_source(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_search_rag_dataset",
            "--train_retrieval_cache",
            "train.json",
            "--test_retrieval_cache",
            "test.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
