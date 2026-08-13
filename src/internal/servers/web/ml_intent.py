"""Lazy adapter: canonical-example index -> RouteStrategy for route_query."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.internal.configs import AppSettings, load_app_settings
from src.internal.servers.web import request_capture as _capture
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_encoder import encode_texts

logger = logging.getLogger(__name__)

_INTENT_INDEXES: dict[Path, object | None] = {}

_ROUTE_VALUES = {s.value for s in RouteStrategy}


@dataclass(frozen=True)
class IntentModelDecision:
    """A valid route prediction with serving diagnostics."""

    strategy: RouteStrategy
    confidence: float
    threshold: float
    latency_ms: float
    modules: tuple[str, ...] = ()


def intent_min_confidence(settings: AppSettings | None = None) -> float:
    """Return the configured similarity threshold.

    This is a cosine similarity, not a softmax probability. The two live on
    different scales, so a value carried over from the previous model is
    meaningless here.
    """
    return (settings or load_app_settings()).intent_model_min_confidence


def load_intent_index(settings: AppSettings | None = None) -> object | None:
    """Load the configured index lazily, caching by resolved path.

    Loading is lazy rather than done at app startup: the web TestClient suite
    already hangs on lifespan model loads, and routing degrades safely to the
    LLM classifier while the encoder warms.
    """
    resolved = settings or load_app_settings()
    configured = resolved.intent_index_path
    if configured is None:
        return None
    directory = configured.resolve()
    if directory in _INTENT_INDEXES:
        return _INTENT_INDEXES[directory]
    try:
        from src.model.intent_knn import INDEX_FILENAME, IntentIndex

        index = IntentIndex.load(directory / INDEX_FILENAME)
    except Exception:
        logger.exception("intent-index: load failed — similarity routing disabled")
        _INTENT_INDEXES[directory] = None
    else:
        low_support = index.low_support_modules()
        if low_support:
            logger.warning(
                "intent-index: modules below support, not emitted: %s",
                ", ".join(low_support),
            )
        _INTENT_INDEXES[directory] = index
    return _INTENT_INDEXES[directory]


def predict_route(
    query: str, *, settings: AppSettings | None = None
) -> IntentModelDecision | None:
    """Return a supported route decision, or None to defer to the classifier.

    Confidence abstention is *not* handled here: the decision is returned and
    ``route_request`` applies its existing confidence-versus-threshold rule
    unchanged. Margin abstention has no equivalent there, so it returns None
    after recording its own capture stage. Both paths end at the LLM classifier.
    """
    resolved = settings or load_app_settings()
    index = load_intent_index(resolved)
    if index is None:
        return None
    start = perf_counter()
    try:
        vector = encode_texts([query])[0]
        decision = index.decide(
            vector,
            min_confidence=resolved.intent_model_min_confidence,
            min_margin=resolved.intent_min_route_margin,
            min_module_score=resolved.intent_min_module_score,
        )
    except Exception:
        logger.exception("intent-index: predict failed — deferring")
        return None
    latency_ms = (perf_counter() - start) * 1_000

    if decision.route not in _ROUTE_VALUES:
        return None
    confidence = float(decision.confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        logger.warning("intent-index: invalid confidence — deferring")
        return None

    if decision.abstain_reason == "margin_below_threshold":
        _capture.record_stage(
            "intent_model",
            "evaluation",
            {
                "predicted_intent": decision.route,
                "confidence": confidence,
                "margin": float(decision.margin),
                "abstained": True,
                "fallback_reason": "margin_below_threshold",
                "composite": decision.composite,
                "latency_ms": latency_ms,
            },
        )
        return None

    return IntentModelDecision(
        strategy=RouteStrategy(decision.route),
        confidence=confidence,
        threshold=resolved.intent_model_min_confidence,
        latency_ms=latency_ms,
        modules=decision.modules,
    )
