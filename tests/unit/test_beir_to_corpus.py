"""Tests for the BEIR -> demo corpus converter.

`beir` is not a test dependency, so the download/load half is stubbed. What is
actually under test is the wiring: a converted corpus must end up *registered*,
because writing the .jsonl alone does not make `--corpus <name>` resolve.
"""

from __future__ import annotations

import json

import pytest

from examples import beir_to_corpus
from src.internal.servers.retrieval.corpus_registry import (
    load_manifest,
    resolve_corpus_docs,
)

_CORPUS = {
    "d1": {"title": "Vitamin C", "text": "Ascorbic acid in citrus fruit."},
    "d2": {"title": "Zinc", "text": "Zinc supplementation and colds."},
}


@pytest.fixture
def stub_beir(monkeypatch):
    monkeypatch.setattr(
        beir_to_corpus, "_load_beir_corpus", lambda dataset, data_dir: _CORPUS
    )


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", argv)
    beir_to_corpus.main()


def test_converting_a_dataset_registers_it_under_its_name(
    stub_beir, monkeypatch, tmp_path
):
    """The gap this closes: the corpus file was written but never registered,
    so `--corpus nfcorpus` failed until the manifest was hand-edited."""
    _run(monkeypatch, tmp_path, ["beir_to_corpus.py", "--dataset", "nfcorpus"])

    manifest = load_manifest(str(tmp_path / "data" / "corpora.json"))
    assert manifest["nfcorpus"]["path"] == "data/corpus_nfcorpus.jsonl"
    assert manifest["nfcorpus"]["docs"] == 2
    assert manifest["nfcorpus"]["domain"] == "medical/nutrition"
    assert manifest["nfcorpus"]["source"] == "NFCorpus"


def test_the_registered_corpus_actually_resolves_by_name(
    stub_beir, monkeypatch, tmp_path
):
    """End to end: convert, then load by name the way the servers do."""
    _run(monkeypatch, tmp_path, ["beir_to_corpus.py", "--dataset", "nfcorpus"])

    manifest = load_manifest(str(tmp_path / "data" / "corpora.json"))
    docs = resolve_corpus_docs("nfcorpus", manifest=manifest)

    assert sorted(d["id"] for d in docs) == ["d1", "d2"]
    assert docs[0]["metadata"]["source"] == "NFCorpus"


def test_a_custom_out_path_is_the_one_registered(stub_beir, monkeypatch, tmp_path):
    """--out moves the file; the manifest must point at where it went."""
    _run(
        monkeypatch,
        tmp_path,
        ["beir_to_corpus.py", "--dataset", "scifact", "--out", "data/custom.jsonl"],
    )

    manifest = load_manifest(str(tmp_path / "data" / "corpora.json"))
    assert manifest["scifact"]["path"] == "data/custom.jsonl"


def test_registering_does_not_discard_corpora_already_in_the_manifest(
    stub_beir, monkeypatch, tmp_path
):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpora.json").write_text(
        json.dumps({"demo": {"path": "data/corpus.jsonl", "docs": 20}})
    )

    _run(monkeypatch, tmp_path, ["beir_to_corpus.py", "--dataset", "nfcorpus"])

    manifest = load_manifest(str(tmp_path / "data" / "corpora.json"))
    assert set(manifest) == {"demo", "nfcorpus"}


def test_an_unknown_dataset_still_registers_without_invented_metadata(
    stub_beir, monkeypatch, tmp_path
):
    """BEIR has more datasets than SUPPORTED lists; an unlisted one must still
    be usable, and must not be labelled with a domain nobody verified."""
    _run(monkeypatch, tmp_path, ["beir_to_corpus.py", "--dataset", "quora"])

    entry = load_manifest(str(tmp_path / "data" / "corpora.json"))["quora"]
    assert entry["domain"] is None
    assert entry["source"] == "quora"


def test_a_corrupt_manifest_does_not_cost_the_converted_corpus(
    stub_beir, monkeypatch, tmp_path, capsys
):
    """The .jsonl is written before the manifest. A manifest the user broke by
    hand must be reported, not silently overwritten, and must not make the
    conversion look like it failed -- the corpus file is on disk and usable."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpora.json").write_text("{not json")

    _run(monkeypatch, tmp_path, ["beir_to_corpus.py", "--dataset", "nfcorpus"])

    out = capsys.readouterr().out
    assert (tmp_path / "data" / "corpus_nfcorpus.jsonl").exists()
    assert "could not be parsed" in out
    assert "--corpus_path" in out  # tells the user how to use it anyway
    assert (tmp_path / "data" / "corpora.json").read_text() == "{not json"


def test_help_lists_every_supported_dataset_with_its_domain():
    """`SUPPORTED` carries the manifest fields *and* renders the --help epilog.
    Restructuring it for registration must not break the listing."""
    help_text = beir_to_corpus._build_parser().format_help()

    for name, meta in beir_to_corpus.SUPPORTED.items():
        assert f"{name}" in help_text
        assert meta["size"] in help_text
        assert meta["domain"] in help_text
        # The manifest needs a citation label for every dataset it can register.
        assert meta["source"]
