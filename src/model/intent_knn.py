"""Route a request by similarity to curated canonical examples.

The route is the argmax of a per-route score: the mean of the top-3 cosine
similarities among that route's canonical examples. Two thresholds gate the
result, because two different things go wrong. A low absolute score means
nothing canonical resembles the request — out of scope. A low margin between
the best and second-best route means two routes fit equally well — ambiguous.
Either one abstains, and the caller falls through to the LLM classifier.

Cosine is deliberately not normalized across routes. A softmax head sums to one
by construction and so cannot express "none of these", which is why the previous
model's out-of-scope separation was only +0.059.

This module imports numpy and nothing heavier: all routing math must run in a
CI job that installs neither torch nor sentence-transformers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .intent_taxonomy import (
    ACTION_MODULES,
    INTENT_LABELS,
    modules_for_route,
    validate_modules,
)

TOP_K = 3
MIN_MODULE_SUPPORT = 10
INDEX_FILENAME = "index.npz"

_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CanonicalExample:
    """One curated routing example: the unit of the index."""

    id: str
    text: str
    route: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class KnnDecision:
    """A route decision with its diagnostics and, if any, its abstention."""

    route: str
    confidence: float
    margin: float
    modules: tuple[str, ...]
    composite: bool
    abstained: bool
    abstain_reason: str | None


class IntentIndex:
    """Canonical example vectors and the scoring rules over them."""

    def __init__(
        self,
        examples: Sequence[CanonicalExample],
        vectors: np.ndarray,
        encoder: str,
        fingerprint: str,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(examples):
            raise ValueError(
                "Index rows must equal the example count: "
                f"{vectors.shape} against {len(examples)} examples"
            )
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise ValueError(
                "Index vectors must be L2-normalized; cosine similarity is "
                "otherwise an arbitrary dot product with no later symptom"
            )
        for example in examples:
            validate_modules(example.route, example.modules)

        self._examples = tuple(examples)
        self._vectors = vectors.astype(np.float32)
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._route_rows = {
            route: np.array(
                [i for i, e in enumerate(self._examples) if e.route == route],
                dtype=np.int64,
            )
            for route in INTENT_LABELS
        }
        self._module_rows = {
            module: np.array(
                [i for i, e in enumerate(self._examples) if module in e.modules],
                dtype=np.int64,
            )
            for module in (
                m for route in INTENT_LABELS for m in modules_for_route(route)
            )
        }

    @property
    def size(self) -> int:
        return len(self._examples)

    @property
    def encoder(self) -> str:
        return self._encoder

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def examples(self) -> tuple[CanonicalExample, ...]:
        return self._examples

    @property
    def vectors(self) -> np.ndarray:
        """The normalized example matrix, for callers that need raw similarity."""
        return self._vectors

    def _top_k_mean(self, similarities: np.ndarray, rows: np.ndarray) -> float:
        """Mean of the highest TOP_K similarities among *rows*, or 0.0 if empty.

        Taking a top-k mean rather than the mean of all rows keeps one close
        neighbor from being diluted by distant same-route examples, and keeps
        one outlier from carrying the route the way a bare max would.
        """
        if rows.size == 0:
            return 0.0
        selected = similarities[rows]
        if selected.size > TOP_K:
            selected = np.partition(selected, -TOP_K)[-TOP_K:]
        return float(selected.mean())

    def _similarities(self, vector: np.ndarray) -> np.ndarray:
        return self._vectors @ np.asarray(vector, dtype=np.float32)

    def route_scores(self, vector: np.ndarray) -> dict[str, float]:
        """Per-route top-3 mean cosine."""
        similarities = self._similarities(vector)
        return {
            route: self._top_k_mean(similarities, rows)
            for route, rows in self._route_rows.items()
        }

    def module_scores(self, vector: np.ndarray) -> dict[str, float]:
        """Per-module top-3 mean cosine, over every module in the taxonomy."""
        similarities = self._similarities(vector)
        return {
            module: self._top_k_mean(similarities, rows)
            for module, rows in self._module_rows.items()
        }

    def low_support_modules(self, minimum: int = MIN_MODULE_SUPPORT) -> tuple[str, ...]:
        """Modules with too few examples to score meaningfully."""
        return tuple(
            module for module, rows in self._module_rows.items() if rows.size < minimum
        )

    def decide(
        self,
        vector: np.ndarray,
        *,
        min_confidence: float,
        min_margin: float,
        min_module_score: float,
    ) -> KnnDecision:
        """Score *vector*, apply both routing thresholds, and report modules."""
        routes = self.route_scores(vector)
        ranked = sorted(routes.items(), key=lambda item: item[1], reverse=True)
        (best_route, confidence), (runner_up, runner_up_score) = ranked[0], ranked[1]
        margin = confidence - runner_up_score

        # Confidence is checked first: a request far from everything is out of
        # scope, which is a more useful thing to say than "ambiguous".
        abstain_reason: str | None = None
        if confidence < min_confidence:
            abstain_reason = "confidence_below_threshold"
        elif margin < min_margin:
            abstain_reason = "margin_below_threshold"

        modules = self._emit_modules(vector, best_route, min_module_score)
        return KnnDecision(
            route=best_route,
            confidence=confidence,
            margin=margin,
            modules=modules,
            composite=self._is_composite(vector, runner_up, margin, min_margin),
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
        )

    def _emit_modules(
        self, vector: np.ndarray, route: str, min_module_score: float
    ) -> tuple[str, ...]:
        """Every well-supported module of *route* scoring at or above the bar.

        Falls back to the single best so a decision always carries a module.
        This is multi-label because real requests carry several intents at once:
        "compare the current prices of BTC and ETH" is both current_info and
        lookup_fact.
        """
        low_support = set(self.low_support_modules())
        scores = self.module_scores(vector)
        # A module with zero examples has no real score at all (module_scores
        # reports 0.0 for it by construction) and must never be a candidate,
        # even as a fallback. "Low support" (< MIN_MODULE_SUPPORT) is a softer,
        # ignorable-under-fallback signal for modules that do have examples.
        populated = {
            module: scores[module]
            for module in modules_for_route(route)
            if self._module_rows[module].size > 0
        }
        candidates = {
            module: score
            for module, score in populated.items()
            if module not in low_support
        }
        if not candidates:
            candidates = populated
        emitted = tuple(
            module for module, score in candidates.items() if score >= min_module_score
        )
        if emitted:
            return emitted
        return (max(candidates, key=lambda module: candidates[module]),)

    def _is_composite(
        self, vector: np.ndarray, runner_up: str, margin: float, min_margin: float
    ) -> bool:
        """True when a close runner-up route is an action.

        This is the signature of a request that needs two steps — "find the best
        Italian place near the office and book it for 7". Nothing acts on the
        flag yet; it is recorded so the plan-aware router can be designed
        against measured data rather than guesses.
        """
        if margin >= min_margin:
            return False
        scores = self.module_scores(vector)
        candidates = modules_for_route(runner_up)
        if not candidates:
            return False
        best = max(candidates, key=lambda module: scores[module])
        return best in ACTION_MODULES

    def save(self, path: Path) -> None:
        """Write vectors and labels to a single npz."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self._vectors,
            examples=np.array(
                json.dumps(
                    [
                        {
                            "id": e.id,
                            "text": e.text,
                            "route": e.route,
                            "modules": list(e.modules),
                        }
                        for e in self._examples
                    ]
                )
            ),
            encoder=np.array(self._encoder),
            fingerprint=np.array(self._fingerprint),
        )

    @classmethod
    def load(cls, path: Path) -> "IntentIndex":
        """Read an index written by ``save``."""
        if not path.exists():
            raise FileNotFoundError(
                f"Intent index is missing {path.name}: {path}. Run "
                "`python -m src.model.intent_index_cli build` to create it."
            )
        payload = np.load(path, allow_pickle=False)
        records = json.loads(str(payload["examples"]))
        examples = [
            CanonicalExample(
                id=record["id"],
                text=record["text"],
                route=record["route"],
                modules=tuple(record["modules"]),
            )
            for record in records
        ]
        return cls(
            examples=examples,
            vectors=payload["vectors"],
            encoder=str(payload["encoder"]),
            fingerprint=str(payload["fingerprint"]),
        )
