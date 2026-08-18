"""Lazy adapter: canonical-example index -> RouteStrategy for route_query."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.internal.configs import AppSettings, load_app_settings
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent.model import DEFAULT_ENCODER, encode_texts

logger = logging.getLogger(__name__)

_INTENT_INDEXES: dict[Path, object | None] = {}

_ROUTE_VALUES = {s.value for s in RouteStrategy}


@dataclass(frozen=True)
class IntentModelDecision:
    """A valid route prediction with serving diagnostics.

    ``abstain_reason`` is ``margin_below_threshold`` or ``None``. It was once
    one of two abstentions; the confidence gate that provided the other was
    removed after measuring at 3 changed decisions in 416.

    The optional fields default, so a construction without them means "served"
    — which is what every existing caller and test double intends.
    """

    strategy: RouteStrategy
    confidence: float
    latency_ms: float
    modules: tuple[str, ...] = ()
    composite: bool = False
    margin: float = 0.0
    abstain_reason: str | None = None


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
        from src.model.intent.model import INDEX_FILENAME, IntentIndex

        index = IntentIndex.load(directory / INDEX_FILENAME)
        if index.encoder != DEFAULT_ENCODER:
            # A dimension mismatch would raise inside index.decide(); a same-
            # dimension different model (e.g. MiniLM-L6 vs MiniLM-L12, both
            # 384-d) would not — it would just score silent garbage dot
            # products. Catch both the same way: fail loudly here, once.
            raise ValueError(
                f"intent-index built with encoder {index.encoder!r}, "
                f"serving uses {DEFAULT_ENCODER!r}"
            )
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
    """Return a route decision, or None when there is nothing to say at all.

    Abstention is not decided here. Margin abstention is reported on the
    returned decision's ``abstain_reason`` for ``route_request`` to act on, and
    it ends at the LLM classifier.

    There used to be a second, confidence-based abstention judged by
    ``route_request`` against a configured floor. That gate was measured to
    change 3 decisions out of 416 and removed, so the margin is now the only
    thing that abstains.

    Margin abstention used to return ``None`` after recording its own capture
    stage, which is why it was invisible to production telemetry: ``None`` is
    indistinguishable from "no index configured", so ``route_request`` could
    not tell a deferral from an absent model. Returning the decision makes the
    reporting single-owner.

    This function records **no** capture stage. ``route_request`` records
    exactly one per decision; recording here as well would emit two stages with
    conflicting payloads for any request that abstains.

    ``None`` now means only: no index, an unsupported route, a failed encode, or
    a confidence outside the valid cosine range.
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
            min_margin=resolved.intent_min_route_margin,
            min_module_score=resolved.intent_min_module_score,
            top_k=resolved.intent_top_k,
        )
    except Exception:
        logger.exception("intent-index: predict failed — deferring")
        return None
    latency_ms = (perf_counter() - start) * 1_000

    if decision.route not in _ROUTE_VALUES:
        logger.warning("intent-index: unsupported route %r — deferring", decision.route)
        return None
    confidence = float(decision.confidence)
    if not math.isfinite(confidence) or not -1.0 <= confidence <= 1.0:
        logger.warning(
            "intent-index: non-finite or out-of-range cosine confidence — deferring"
        )
        return None

    return IntentModelDecision(
        strategy=RouteStrategy(decision.route),
        confidence=confidence,
        latency_ms=latency_ms,
        modules=decision.modules,
        composite=decision.composite,
        margin=float(decision.margin),
        abstain_reason=(
            "margin_below_threshold"
            if decision.abstain_reason == "margin_below_threshold"
            else None
        ),
    )
