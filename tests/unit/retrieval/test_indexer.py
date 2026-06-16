"""Tests for FAISS index builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.internal.retrieval.indexer import IndexerConfig, build_faiss_index


def _write_corpus(docs: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        return f.name


def test_build_faiss_index_creates_file():
    corpus = [
        {"id": "d1", "title": "T1", "contents": "text one"},
        {"id": "d2", "title": "T2", "contents": "text two"},
    ]
    corpus_path = _write_corpus(corpus)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(2, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(
                    corpus_path=corpus_path,
                    index_path=index_path,
                    model_name="intfloat/e5-base-v2",
                )
            )
        assert Path(index_path).exists()


def test_build_faiss_index_stores_correct_count():
    import faiss

    corpus = [
        {"id": f"d{i}", "title": f"T{i}", "contents": f"text {i}"} for i in range(5)
    ]
    corpus_path = _write_corpus(corpus)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(5, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(corpus_path=corpus_path, index_path=index_path)
            )
        index = faiss.read_index(index_path)
        assert index.ntotal == 5


def test_build_faiss_index_uses_hnsw_with_correct_params():
    import faiss

    corpus = [{"id": "d1", "contents": "text"}]
    corpus_path = _write_corpus(corpus)

    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(1, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(
                    corpus_path=corpus_path,
                    index_path=index_path,
                    ef_construction=128,
                    ef_search=64,
                )
            )
        index = faiss.read_index(index_path)
        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.hnsw.efSearch == 64


def test_build_faiss_index_empty_corpus():
    corpus_path = _write_corpus([])
    fake_embedder = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        index_path = str(Path(tmp) / "test.index")
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(corpus_path=corpus_path, index_path=index_path)
            )
        # Empty corpus → no index file written
        assert not Path(index_path).exists()
