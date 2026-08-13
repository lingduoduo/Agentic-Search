from pathlib import Path

import numpy as np
import pytest

from src.model.intent_pretrained import (
    EMBEDDINGS_FILENAME,
    VOCAB_FILENAME,
    load_pretrained_bundle,
    write_pretrained_bundle,
)

_TOKENS = (
    ["[PAD]"] + [f"[unused{i}]" for i in range(99)] + ["[UNK]", "the", "dashboard"]
)


def _write(directory: Path, *, tokens=None, rows: int | None = None, dim: int = 4):
    tokens = list(_TOKENS if tokens is None else tokens)
    matrix = np.arange(
        (rows if rows is not None else len(tokens)) * dim, dtype=np.float16
    ).reshape(-1, dim)
    write_pretrained_bundle(directory, tokens=tokens, embeddings=matrix)
    return matrix


def test_round_trip_preserves_vocabulary_and_matrix(tmp_path: Path):
    written = _write(tmp_path)

    bundle = load_pretrained_bundle(tmp_path)

    assert bundle.size == len(_TOKENS)
    assert bundle.dim == 4
    assert bundle.embeddings.dtype == np.float16
    np.testing.assert_array_equal(bundle.embeddings, written)
    assert bundle.vocabulary.encode("dashboard") == [_TOKENS.index("dashboard")]


def test_written_vocabulary_is_one_token_per_line_in_id_order(tmp_path: Path):
    _write(tmp_path)

    lines = (tmp_path / VOCAB_FILENAME).read_text(encoding="utf-8").splitlines()

    assert lines == _TOKENS
    assert lines[0] == "[PAD]"


def test_load_rejects_a_matrix_whose_rows_disagree_with_the_vocabulary(
    tmp_path: Path,
):
    """A shifted vocabulary gives every token the wrong vector, silently."""
    _write(tmp_path, rows=len(_TOKENS) - 1)

    with pytest.raises(ValueError, match="rows"):
        load_pretrained_bundle(tmp_path)


def test_load_reports_a_missing_bundle_by_name(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=EMBEDDINGS_FILENAME):
        load_pretrained_bundle(tmp_path)


def test_write_rejects_a_non_float16_matrix(tmp_path: Path):
    with pytest.raises(ValueError, match="float16"):
        write_pretrained_bundle(
            tmp_path,
            tokens=_TOKENS,
            embeddings=np.zeros((len(_TOKENS), 4), dtype=np.float32),
        )
