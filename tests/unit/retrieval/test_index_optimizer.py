"""Tests for FAISSIndexBuilder and HNSWTuner."""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # noqa: E402

from src.internal.retrieval.index_optimizer import (  # noqa: E402
    FAISSIndexBuilder,
    FaissIndexConfig,
    HNSWTuner,
)


def _random_vecs(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, d)).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def test_build_ivfpq_returns_trained_index():
    vecs = _random_vecs(2000, 64)
    builder = FAISSIndexBuilder(FaissIndexConfig(nlist=16, m=8, nbits=8, nprobe=4))
    index = builder.build_ivfpq(vecs, training_sample=500)
    assert index.is_trained
    assert index.ntotal == 2000


def test_build_ivfpq_search_returns_k_results():
    vecs = _random_vecs(500, 64)
    builder = FAISSIndexBuilder(FaissIndexConfig(nlist=8, m=8, nbits=8, nprobe=2))
    index = builder.build_ivfpq(vecs, training_sample=300)
    query = _random_vecs(1, 64)
    _D, indices = index.search(query, 5)
    assert indices.shape == (1, 5)
    assert all(i >= 0 for i in indices[0])


def test_config_defaults():
    cfg = FaissIndexConfig()
    assert cfg.nlist == 4096
    assert cfg.m == 96
    assert cfg.nbits == 8
    assert cfg.nprobe == 64


def test_hnsw_tuner_returns_int():
    vecs = _random_vecs(200, 32)
    index = faiss.IndexHNSWFlat(32, 16)
    index.add(vecs)

    def embedder(q):
        return _random_vecs(1, 32)[0]

    qa_pairs = [{"query": "q", "relevant_doc_ids": [str(i) for i in range(5)]}]

    tuner = HNSWTuner()
    result = tuner.calibrate(index, qa_pairs, embedder, target_recall_at_10=0.0)
    assert isinstance(result, int)
    assert result > 0
