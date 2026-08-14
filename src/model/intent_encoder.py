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

DEFAULT_ENCODER = "intfloat/e5-small-v2"

# E5 models are trained with instruction prefixes and degrade *silently*
# without them — no error, just worse vectors. The prefix is therefore a
# property of the model, derived from its name rather than passed by callers,
# so no call site can omit it. Deriving rather than storing it also means the
# index needs no new field: it already records the encoder name, and
# ml_intent.load_intent_index already rejects a mismatch, which covers the
# prefix for free. That matters here because e5-small-v2 is also 384-wide, so
# an index built with the previous encoder would otherwise load and score
# without any error at all.
#
# Both sides of the comparison use "query: ": this is symmetric short-text
# similarity, not the asymmetric query/passage retrieval "passage: " is for.
MODEL_PREFIXES: dict[str, str] = {
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}


def prefix_for(model_name: str) -> str:
    """The instruction prefix *model_name* requires.

    Raises rather than defaulting to "": an unregistered model is far more
    likely to be one whose prefix nobody looked up than one that genuinely
    needs none, and guessing wrong is invisible.
    """
    try:
        return MODEL_PREFIXES[model_name]
    except KeyError:
        raise ValueError(
            f"No instruction prefix registered for encoder {model_name!r}. "
            f"Add it to MODEL_PREFIXES — encoders that need a prefix degrade "
            f"silently without one. Known: {sorted(MODEL_PREFIXES)}"
        ) from None


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
    prefix = prefix_for(model_name)
    vectors = _model(model_name).encode(
        [prefix + text for text in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
