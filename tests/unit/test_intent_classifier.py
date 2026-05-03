"""Unit tests for src.agent_loop.intent_classifier and intent routing helpers."""

from src.agent_loop.intent_classifier import (
    INTENT_LABELS,
    IntentPipeline,
    IntentPrediction,
    IntentionClassificationPipeline,
    resolve_search_settings,
)
from src.search.vocabulary import Vocabulary


def test_vocabulary_build_and_encode_support_sequence_training():
    vocab = Vocabulary()
    vocab.build([["buy", "phone"], ["buy", "laptop"]], min_freq=1)

    encoded = vocab.encode(["buy", "unknown", "phone"])

    assert vocab.token2cnt["buy"] == 2
    assert encoded[0] != 0
    assert encoded[1] == 0


def test_pipeline_predict_requires_training():
    pipeline = IntentPipeline()
    try:
        pipeline.predict(["buy", "phone"])
    except RuntimeError as exc:
        assert "not trained" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("predict() should require training first")


def test_intention_pipeline_alias_matches_intent_pipeline():
    assert IntentionClassificationPipeline is IntentPipeline


def test_resolve_search_settings_purchase_forces_search_bias():
    topk, max_sl, req_ev, allow_int, meta = resolve_search_settings(
        IntentPrediction(intent="purchase", confidence=0.95),
        topk=3,
        max_search_limit=1,
        require_evidence=False,
        allow_internal_knowledge=True,
        min_confidence=0.6,
    )

    assert topk == 8
    assert max_sl == 2
    assert req_ev is True
    assert allow_int is False
    assert meta["predicted_intent"] == "purchase"
    assert meta["intent_policy_applied"] is True


def test_resolve_search_settings_low_confidence_keeps_defaults():
    topk, max_sl, req_ev, allow_int, meta = resolve_search_settings(
        IntentPrediction(intent="qa", confidence=0.2),
        topk=5,
        max_search_limit=4,
        require_evidence=True,
        allow_internal_knowledge=True,
        min_confidence=0.6,
    )

    assert topk == 5
    assert max_sl == 4
    assert req_ev is True
    assert allow_int is True
    assert meta["intent_policy_applied"] is False


def test_resolve_search_settings_qa_keeps_defaults():
    topk, max_sl, req_ev, allow_int, meta = resolve_search_settings(
        IntentPrediction(intent="qa", confidence=0.9),
        topk=5,
        max_search_limit=4,
        require_evidence=True,
        allow_internal_knowledge=True,
    )

    assert topk == 5
    assert max_sl == 4
    assert meta["intent_policy_applied"] is True


def test_resolve_search_settings_recommendation_boosts_results():
    topk, max_sl, *_ = resolve_search_settings(
        IntentPrediction(intent="recommendation", confidence=0.85),
        topk=3,
        max_search_limit=1,
        require_evidence=False,
        allow_internal_knowledge=True,
    )

    assert topk == 8
    assert max_sl == 3


def test_intent_labels_snapshot():
    assert INTENT_LABELS == ["purchase", "navigate", "qa", "recommendation"]
