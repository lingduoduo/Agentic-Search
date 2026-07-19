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
    """Two corpora with disjoint id namespaces (the expected, mergeable case)."""
    _write_corpus(
        tmp_path / "a.jsonl", [{"id": "a1", "title": "A1", "contents": "alpha"}]
    )
    _write_corpus(
        tmp_path / "b.jsonl",
        [
            {"id": "b1", "title": "B1", "contents": "beta"},
            {"id": "b2", "title": "B2", "contents": "gamma"},
        ],
    )
    return {
        "a": {"path": str(tmp_path / "a.jsonl")},
        "b": {"path": str(tmp_path / "b.jsonl")},
    }


def test_resolve_by_name(manifest):
    docs = resolve_corpus_docs("a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_all_unions_disjoint_corpora(manifest):
    docs = resolve_corpus_docs("all", manifest)
    assert [d["id"] for d in docs] == ["a1", "b1", "b2"]


def test_resolve_comma_list_preserves_order(manifest):
    docs = resolve_corpus_docs("b,a", manifest)
    assert [d["id"] for d in docs] == ["b1", "b2", "a1"]


def test_resolve_same_corpus_repeated_dedupes(manifest):
    # A registered corpus loaded twice collapses its own repeat (no collision).
    docs = resolve_corpus_docs("a,a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_cross_corpus_id_collision_raises(tmp_path):
    # Two *different* corpora sharing an id must fail loud, not silently drop.
    _write_corpus(tmp_path / "x.jsonl", [{"id": "5", "title": "X", "contents": "x"}])
    _write_corpus(tmp_path / "y.jsonl", [{"id": "5", "title": "Y", "contents": "y"}])
    colliding = {
        "x": {"path": str(tmp_path / "x.jsonl")},
        "y": {"path": str(tmp_path / "y.jsonl")},
    }
    with pytest.raises(ValueError, match="id collision"):
        resolve_corpus_docs("all", colliding)


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
    rows = retriever.retrieve(["gamma"], topk=5)
    assert rows[0][0]["document"]["id"] == "b2"


def test_resolve_all_skips_missing_files_and_loads_present(manifest, caplog):
    manifest = dict(manifest)
    manifest["missing"] = {"path": "/nonexistent/path/does-not-exist.jsonl"}
    with caplog.at_level("WARNING"):
        docs = resolve_corpus_docs("all", manifest)
    assert [d["id"] for d in docs] == ["a1", "b1", "b2"]
    assert "missing" in caplog.text
    assert "/nonexistent/path/does-not-exist.jsonl" in caplog.text


def test_resolve_missing_named_corpus_raises(manifest):
    manifest = dict(manifest)
    manifest["missing"] = {"path": "/nonexistent/path/does-not-exist.jsonl"}
    with pytest.raises(ValueError):
        resolve_corpus_docs("missing", manifest)


def test_resolve_empty_spec_raises(manifest):
    with pytest.raises(ValueError):
        resolve_corpus_docs("", manifest)
    with pytest.raises(ValueError):
        resolve_corpus_docs(None, manifest)
