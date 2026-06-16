"""Unit tests for ChunkConfig — validates env-var defaults and bounds."""

from __future__ import annotations

import pytest
from src.internal.retrieval.chunk_config import ChunkConfig


def test_defaults():
    cfg = ChunkConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_from_env_reads_vars(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")
    cfg = ChunkConfig.from_env()
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=64, chunk_overlap=64)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkConfig(chunk_size=0)


def test_chunk_overlap_must_be_nonnegative():
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkConfig(chunk_size=512, chunk_overlap=-1)
