"""Lazy adapter: trained intent classifier -> RouteStrategy for route_query."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

from src.internal.configs import AppSettings, load_app_settings
from src.internal.servers.web.intent_routing import RouteStrategy

logger = logging.getLogger(__name__)

_INTENT_MODELS: dict[Path, object | None] = {}

_ROUTE_VALUES = {s.value for s in RouteStrategy}


@dataclass(frozen=True)
class IntentModelDecision:
    """A valid route prediction with serving diagnostics."""

    strategy: RouteStrategy
    confidence: float
    threshold: float
    latency_ms: float


def intent_min_confidence(settings: AppSettings | None = None) -> float:
    """Return the configured model confidence threshold."""
    return (settings or load_app_settings()).intent_model_min_confidence


def load_intent_model(settings: AppSettings | None = None) -> object | None:
    """Load the configured intent model lazily, caching by resolved artifact."""
    resolved_settings = settings or load_app_settings()
    configured_path = resolved_settings.intent_model_path
    if configured_path is None:
        return None
    artifact_path = configured_path.resolve()
    if artifact_path in _INTENT_MODELS:
        return _INTENT_MODELS[artifact_path]
    try:
        from src.model.intent_classifier import IntentPipeline

        model = IntentPipeline.load(str(artifact_path))
    except Exception:
        logger.exception("intent-model: load failed — ML routing disabled")
        _INTENT_MODELS[artifact_path] = None
    else:
        _INTENT_MODELS[artifact_path] = model
    return _INTENT_MODELS[artifact_path]


def predict_route(
    query: str, *, settings: AppSettings | None = None
) -> IntentModelDecision | None:
    """Return a supported model route decision, or None to defer."""
    resolved_settings = settings or load_app_settings()
    model = load_intent_model(resolved_settings)
    if model is None:
        return None
    start = perf_counter()
    try:
        pred = model.predict_text(query)
    except Exception:
        logger.exception("intent-model: predict failed — deferring")
        return None
    if pred.intent not in _ROUTE_VALUES:
        return None
    return IntentModelDecision(
        strategy=RouteStrategy(pred.intent),
        confidence=float(pred.confidence),
        threshold=resolved_settings.intent_model_min_confidence,
        latency_ms=(perf_counter() - start) * 1_000,
    )
