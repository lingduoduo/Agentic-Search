"""Tests for FAISS index builder.

Uses sys.modules patching to mock faiss — no faiss installation required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

from src.internal.retrieval.indexer import IndexerConfig, build_faiss_index


def _write_corpus(docs: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        return f.name


@contextmanager
def _fake_faiss():
    """Inject a MagicMock faiss into sys.modules for the duration of the block."""
    mock_faiss = MagicMock(name="faiss")
    fake_index = MagicMock(name="faiss.Index")
    mock_faiss.IndexHNSWFlat.return_value = fake_index
    with patch.dict(sys.modules, {"faiss": mock_faiss}):
        yield mock_faiss, fake_index


def test_build_faiss_index_calls_write_index():
    corpus = [
        {"id": "d1", "title": "T1", "contents": "text one"},
        {"id": "d2", "title": "T2", "contents": "text two"},
    ]
    corpus_path = _write_corpus(corpus)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(2, 768).astype(np.float32)

    with _fake_faiss() as (mock_faiss, fake_index):
        index_path = "/tmp/test.index"
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
        mock_faiss.write_index.assert_called_once_with(fake_index, index_path)


def test_build_faiss_index_adds_correct_count():
    corpus = [{"id": f"d{i}", "contents": f"text {i}"} for i in range(5)]
    corpus_path = _write_corpus(corpus)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(5, 768).astype(np.float32)

    with _fake_faiss() as (_mock_faiss, fake_index):
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(corpus_path=corpus_path, index_path="/tmp/x.index")
            )

        # index.add() receives all 5 embedding vectors
        add_call_arg = fake_index.add.call_args[0][0]
        assert add_call_arg.shape[0] == 5


def test_build_faiss_index_sets_hnsw_params():
    corpus = [{"id": "d1", "contents": "text"}]
    corpus_path = _write_corpus(corpus)
    fake_embedder = MagicMock()
    fake_embedder.encode.return_value = np.random.randn(1, 768).astype(np.float32)

    with _fake_faiss() as (mock_faiss, fake_index):
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(
                    corpus_path=corpus_path,
                    index_path="/tmp/x.index",
                    ef_construction=128,
                    ef_search=64,
                    hnsw_m=32,
                )
            )

        # IndexHNSWFlat called with (dim=768, M=32)
        mock_faiss.IndexHNSWFlat.assert_called_once_with(768, 32)
        # efConstruction and efSearch assigned on the index
        assert fake_index.hnsw.efConstruction == 128
        assert fake_index.hnsw.efSearch == 64


def test_build_faiss_index_skips_write_on_empty_corpus():
    corpus_path = _write_corpus([])
    fake_embedder = MagicMock()

    with _fake_faiss() as (mock_faiss, _fake_index):
        with patch(
            "src.internal.retrieval.indexer._load_embedder", return_value=fake_embedder
        ):
            build_faiss_index(
                IndexerConfig(corpus_path=corpus_path, index_path="/tmp/x.index")
            )

        # Empty corpus → embedder never called, write_index never called
        fake_embedder.encode.assert_not_called()
        mock_faiss.write_index.assert_not_called()
