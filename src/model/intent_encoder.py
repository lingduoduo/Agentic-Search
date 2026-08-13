"""The only place a sentence encoder is loaded.

The import is function-local on purpose. Every other intent module must stay
importable in a CI job that installs neither torch nor sentence-transformers,
and this repo has twice shipped collection failures from unguarded imports
(#356, re-fixed in #418). Keeping the dependency behind one function is what
makes the rest of the routing path testable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _model(model_name: str):
    """Load and cache the encoder. Loading costs seconds; encoding costs ms."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def encode_texts(
    texts: Sequence[str], *, model_name: str = DEFAULT_ENCODER
) -> np.ndarray:
    """Encode *texts* to L2-normalized float32 rows.

    Normalizing here means every consumer can treat a dot product as a cosine,
    and the index constructor can reject anything that is not normalized.
    """
    vectors = _model(model_name).encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encoder_dimension(model_name: str = DEFAULT_ENCODER) -> int:
    """The encoder's output width."""
    return int(_model(model_name).get_sentence_embedding_dimension())
