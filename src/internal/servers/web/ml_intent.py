"""Lazy adapter: trained intent classifier -> RouteStrategy for route_query."""

from __future__ import annotations

import logging
import os

from src.internal.servers.web.intent_routing import RouteStrategy

logger = logging.getLogger(__name__)

_INTENT_MODEL: object | None = None  # None=unset, False=failed/absent, pipeline=loaded

_ROUTE_VALUES = {s.value for s in RouteStrategy}


def intent_min_confidence() -> float:
    return float(os.environ.get("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.6"))


def load_intent_model():
    """Lazy singleton trained intent classifier; None when unavailable."""
    global _INTENT_MODEL
    if _INTENT_MODEL is not None:
        return _INTENT_MODEL or None
    path = os.environ.get("AGENTIC_SEARCH_INTENT_MODEL_PATH", "").strip()
    if not path:
        _INTENT_MODEL = False
        return None
    try:
        from src.model.intent_classifier import IntentPipeline

        _INTENT_MODEL = IntentPipeline.load(path)
    except Exception:
        logger.exception("intent-model: load failed — ML routing disabled")
        _INTENT_MODEL = False
        return None
    return _INTENT_MODEL


def predict_route(query: str) -> "tuple[RouteStrategy, float] | None":
    """(RouteStrategy, confidence) from the trained model, or None to defer."""
    model = load_intent_model()
    if model is None:
        return None
    try:
        pred = model.predict_text(query)
    except Exception:
        logger.exception("intent-model: predict failed — deferring")
        return None
    if pred.intent not in _ROUTE_VALUES:
        return None
    return RouteStrategy(pred.intent), float(pred.confidence)
