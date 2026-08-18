"""The encoder seam. Real encoding is covered where the model is installed."""

import numpy as np
import pytest

from src.model.intent import model as intent_encoder
from src.model.intent.model import DEFAULT_ENCODER


@pytest.fixture(autouse=True)
def _clear_cache():
    intent_encoder._MODEL_CACHE.clear()
    yield
    intent_encoder._MODEL_CACHE.clear()


def test_default_encoder_is_e5_small():
    assert DEFAULT_ENCODER == "intfloat/e5-small-v2"


def test_the_default_encoder_has_a_registered_prefix():
    from src.model.intent.model import DEFAULT_ENCODER, prefix_for

    assert prefix_for(DEFAULT_ENCODER) == "query: "


def test_an_unregistered_model_raises_rather_than_using_no_prefix():
    """A silently missing prefix degrades e5 vectors without erroring."""
    from src.model.intent.model import prefix_for

    with pytest.raises(ValueError, match="prefix"):
        prefix_for("some/unregistered-model")


def test_minilm_is_still_registered_with_an_empty_prefix():
    """Old indexes are rejected by name, but the mapping must stay honest."""
    from src.model.intent.model import prefix_for

    assert prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_encode_applies_the_prefix():
    """The whole contract: encode_texts('x') must equal raw encode('query: x')."""
    pytest.importorskip("sentence_transformers")
    import numpy as np

    from src.model.intent.model import DEFAULT_ENCODER, _model, encode_texts

    through_seam = encode_texts(["find the runbook"])
    raw = _model(DEFAULT_ENCODER).encode(
        ["query: find the runbook"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    np.testing.assert_allclose(through_seam, raw.astype(np.float32), atol=1e-5)


def test_the_prefix_actually_changes_the_vector():
    """Proves the assertion above is not vacuous."""
    pytest.importorskip("sentence_transformers")

    from src.model.intent.model import DEFAULT_ENCODER, _model, encode_texts

    unprefixed = _model(DEFAULT_ENCODER).encode(
        ["find the runbook"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    assert float(encode_texts(["find the runbook"])[0] @ unprefixed[0]) < 0.999


def test_encoded_width_is_unchanged_at_384():
    """Same width as the previous encoder, so index.npz's format is unchanged."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent.model import encode_texts

    assert encode_texts(["find the runbook"]).shape == (1, 384)


def test_encode_returns_normalized_float32_rows():
    pytest.importorskip("sentence_transformers")
    from src.model.intent.model import encode_texts

    vectors = encode_texts(["find the runbook", "send an email to the team"])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic():
    """Routing must not flip between requests for a fixed query."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent.model import encode_texts

    first = encode_texts(["find the runbook"])
    second = encode_texts(["find the runbook"])

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_word_order_changes_the_vector():
    """The whole point of the encoder: a bag of embeddings could not do this."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent.model import encode_texts

    vectors = encode_texts(["how to send an email", "send an email to how"])

    assert float(vectors[0] @ vectors[1]) < 0.99


def test_a_failed_load_is_not_retried_on_a_later_call(monkeypatch):
    """A broken or unreachable model must not re-attempt the download per call.

    The first call raises straight from the constructor; the second call must
    raise too, without invoking the constructor again — otherwise a caller on
    the event loop (route_request) would block on the same failing download
    once per auto-routed request instead of degrading once.
    """
    pytest.importorskip("sentence_transformers")
    import sentence_transformers

    calls = {"count": 0}

    def _boom(model_name, device=None):
        calls["count"] += 1
        raise OSError("model fetch failed")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _boom)

    # A registered model name: the constructor is monkeypatched to fail before
    # any real load happens, so this exercises _model's failure caching, not
    # prefix_for's model-name validation.
    broken_model = "intfloat/e5-base-v2"

    with pytest.raises(OSError, match="model fetch failed"):
        intent_encoder.encode_texts(["find the runbook"], model_name=broken_model)

    with pytest.raises(RuntimeError, match="not retrying"):
        intent_encoder.encode_texts(["find the runbook"], model_name=broken_model)

    assert calls["count"] == 1
