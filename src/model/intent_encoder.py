"""The only place a sentence encoder is loaded.

The import is function-local on purpose. Every other intent module must stay
importable in a CI job that installs neither torch nor sentence-transformers,
and this repo has twice shipped collection failures from unguarded imports
(#356, re-fixed in #418). Keeping the dependency behind one function is what
makes the rest of the routing path testable without it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Keyed by model name; holds either a loaded model or an Exception. A failed
# load is cached too — lru_cache does not cache exceptions, and retrying a
# broken or unreachable model download on every call would block whichever
# request triggers it, forever. See ml_intent._INTENT_INDEXES for the same
# policy one layer up.
_MODEL_CACHE: dict[str, object] = {}


def _model(model_name: str):
    """Load and cache the encoder. Loading costs seconds; encoding costs ms."""
    cached = _MODEL_CACHE.get(model_name)
    if isinstance(cached, Exception):
        raise RuntimeError(
            f"intent encoder {model_name!r} failed to load previously; not retrying"
        ) from cached
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        _MODEL_CACHE[model_name] = exc
        raise
    _MODEL_CACHE[model_name] = model
    return model


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
