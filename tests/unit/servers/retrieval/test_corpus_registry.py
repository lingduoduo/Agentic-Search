import json

import pytest

from src.internal.servers.retrieval.corpus_registry import (
    load_manifest,
    resolve_corpus_docs,
)
from src.internal.servers.retrieval.demo import TfidfRetriever


def _write_corpus(path, docs):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


@pytest.fixture
def manifest(tmp_path):
    _write_corpus(
        tmp_path / "a.jsonl", [{"id": "a1", "title": "A1", "contents": "alpha"}]
    )
    _write_corpus(
        tmp_path / "b.jsonl",
        [
            {"id": "b1", "title": "B1", "contents": "beta"},
            {"id": "a1", "title": "dup", "contents": "duplicate of a1"},
        ],
    )
    return {
        "a": {"path": str(tmp_path / "a.jsonl")},
        "b": {"path": str(tmp_path / "b.jsonl")},
    }


def test_resolve_by_name(manifest):
    docs = resolve_corpus_docs("a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_all_unions_and_dedupes_by_id(manifest):
    docs = resolve_corpus_docs("all", manifest)
    # a1 from "a" wins; a1 duplicate in "b" is dropped; b1 kept.
    assert [d["id"] for d in docs] == ["a1", "b1"]


def test_resolve_comma_list(manifest):
    docs = resolve_corpus_docs("b,a", manifest)
    # b first (b1, a1-dup), then a's a1 is a dup and dropped.
    assert [d["id"] for d in docs] == ["b1", "a1"]


def test_resolve_path_backcompat(tmp_path, manifest):
    p = tmp_path / "direct.jsonl"
    _write_corpus(p, [{"id": "z", "title": "Z", "contents": "zeta"}])
    docs = resolve_corpus_docs(str(p), manifest)
    assert [d["id"] for d in docs] == ["z"]


def test_resolve_unknown_spec_raises(manifest):
    with pytest.raises(ValueError, match="Unknown corpus spec"):
        resolve_corpus_docs("nope", manifest)


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(str(tmp_path / "absent.json")) == {}


def test_union_docs_feed_from_docs_retriever(manifest):
    docs = resolve_corpus_docs("all", manifest)
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["beta"], topk=5)
    assert rows[0][0]["document"]["id"] == "b1"
