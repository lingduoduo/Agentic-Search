"""Unit tests for src.agent_loop.intent_classifier and intent routing helpers."""

from src.agent_loop.intent_classifier import (
    INTENT_LABELS,
    IntentPrediction,
    IntentionClassificationPipeline,
)
from src.run_agentic_search import _resolve_intent_routed_search_settings
from src.search.vocabulary import Vocabulary


class DummyIntentPipeline:
    def __init__(self, intent: str, confidence: float):
        self.prediction = IntentPrediction(intent=intent, confidence=confidence)

    def predict_text(self, text: str) -> IntentPrediction:
        assert text
        return self.prediction


def test_vocabulary_build_and_encode_support_sequence_training():
    vocab = Vocabulary()
    vocab.build([["buy", "phone"], ["buy", "laptop"]], min_freq=1)

    encoded = vocab.encode(["buy", "unknown", "phone"])

    assert vocab.token2cnt["buy"] == 2
    assert encoded[0] != 0
    assert encoded[1] == 0


def test_intention_pipeline_predict_requires_training():
    pipeline = IntentionClassificationPipeline()

    try:
        pipeline.predict(["buy", "phone"])
    except RuntimeError as exc:
        assert "Model not trained" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("predict() should require training first")


def test_intent_routing_purchase_forces_search_bias():
    topk, max_search_limit, require_evidence, allow_internal, metadata = (
        _resolve_intent_routed_search_settings(
            question="buy a new phone online",
            topk=3,
            max_search_limit=1,
            require_evidence=False,
            allow_internal_knowledge=True,
            intent_pipeline=DummyIntentPipeline("purchase", 0.95),
            intent_min_confidence=0.6,
        )
    )

    assert topk == 8
    assert max_search_limit == 2
    assert require_evidence is True
    assert allow_internal is False
    assert metadata["predicted_intent"] == "purchase"


def test_intent_routing_low_confidence_keeps_user_settings():
    topk, max_search_limit, require_evidence, allow_internal, metadata = (
        _resolve_intent_routed_search_settings(
            question="what is faiss",
            topk=5,
            max_search_limit=4,
            require_evidence=True,
            allow_internal_knowledge=True,
            intent_pipeline=DummyIntentPipeline("qa", 0.2),
            intent_min_confidence=0.6,
        )
    )

    assert topk == 5
    assert max_search_limit == 4
    assert require_evidence is True
    assert allow_internal is True
    assert metadata["intent_policy_applied"] is False


def test_intent_labels_snapshot():
    assert INTENT_LABELS == ["purchase", "navigate", "qa", "recommendation"]
