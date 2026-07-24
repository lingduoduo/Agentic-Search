import pytest

from src.internal.document_index.models import ChunkingConfig


def test_semantic_defaults_are_off():
    c = ChunkingConfig()
    assert c.semantic_chunking is False
    assert c.semantic_breakpoint_percentile == 95.0
    assert c.semantic_buffer_size == 1
    c.validate()  # defaults are valid


@pytest.mark.parametrize("pct", [0.0, 100.0, -1.0, 150.0])
def test_validate_rejects_bad_percentile(pct):
    with pytest.raises(ValueError):
        ChunkingConfig(semantic_breakpoint_percentile=pct).validate()


def test_validate_rejects_bad_buffer():
    with pytest.raises(ValueError):
        ChunkingConfig(semantic_buffer_size=0).validate()
