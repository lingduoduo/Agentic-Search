"""The encoder seam. Real encoding is covered where the model is installed."""

import numpy as np
import pytest

from src.model.intent_encoder import DEFAULT_ENCODER


def test_default_encoder_is_the_minilm_the_index_is_built_with():
    assert DEFAULT_ENCODER == "sentence-transformers/all-MiniLM-L6-v2"


def test_encode_returns_normalized_float32_rows():
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    vectors = encode_texts(["find the runbook", "send an email to the team"])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic():
    """Routing must not flip between requests for a fixed query."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    first = encode_texts(["find the runbook"])
    second = encode_texts(["find the runbook"])

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_word_order_changes_the_vector():
    """The whole point of the encoder: a bag of embeddings could not do this."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    vectors = encode_texts(["how to send an email", "send an email to how"])

    assert float(vectors[0] @ vectors[1]) < 0.99
