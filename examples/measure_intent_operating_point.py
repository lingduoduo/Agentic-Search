"""Compare two intent encoders at the operating point each would actually run at.

The evaluation report scores one encoder at a time, and its headline is *argmax*
accuracy — the route the index would pick if it always answered. Serving does not
work that way: the margin gate abstains, deferring to the LLM classifier, so what
a caller experiences is two numbers the headline cannot show — how much traffic is
answered, and how often those answers are right.

That distinction decided a real question. On the held-out slice, e5-small-v2's
out-of-scope AUC (0.8551) is *worse* than MiniLM's (0.8848), which reads as "more
accurate but less able to tell when to abstain" — an argument against promoting
it. AUC is a threshold-free ranking statistic over the whole score range, while
the margin gate only needs local separation near the decision boundary. Measured
at the boundary, the ordering reverses: e5 makes an order of magnitude fewer wrong
routes than MiniLM, at its own tuned threshold and at matched coverage alike.

Each encoder's threshold is tuned on the **tuning** slice and reported on the
**test** slice, the same discipline the evaluation pipeline uses — a threshold
chosen on the reported queries would flatter whichever encoder it was chosen for.

Run:

    python -m examples.measure_intent_operating_point

Needs sentence-transformers and both models; the e5 one is the serving default and
the MiniLM one is only pulled for this comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.model.intent_data import (
    load_canonical_examples,
    load_intent_eval_queries,
)
from src.model.intent_eval_split import (
    DEFAULT_SEED,
    DEFAULT_SLICE_SIZE,
    split_eval_queries,
)
from src.model.intent_encoder import DEFAULT_ENCODER, encode_texts
from src.model.intent_knn import TOP_K, IntentIndex

PREVIOUS_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Spans both encoders' useful range. e5 compresses cosines into a narrow band, so
# its margins are roughly an order of magnitude smaller than MiniLM's — a grid
# sized for one is meaningless for the other, which is why this covers both.
MARGIN_GRID = (0.0, 0.004, 0.008, 0.012, 0.015, 0.020, 0.030, 0.050)
MIN_COVERAGE = 0.60


@dataclass(frozen=True)
class OperatingPoint:
    """What a caller experiences at one margin threshold."""

    margin: float
    served: int
    total: int
    correct: int

    @property
    def coverage(self) -> float:
        return self.served / self.total if self.total else 0.0

    @property
    def served_accuracy(self) -> float:
        return self.correct / self.served if self.served else 0.0

    @property
    def wrong_routes(self) -> int:
        """Answers that were wrong — the number a misroute's cost multiplies."""
        return self.served - self.correct


def measure(
    index: IntentIndex, vectors, queries: Sequence, margin: float
) -> OperatingPoint:
    """Score *queries* at *margin*, counting only what the gate lets through."""
    served = correct = 0
    for vector, query in zip(vectors, queries):
        decision = index.decide(
            vector,
            min_confidence=0.0,
            min_margin=margin,
            min_module_score=0.45,
            top_k=TOP_K,
        )
        if not decision.abstained:
            served += 1
            correct += decision.route == query.label
    return OperatingPoint(margin, served, len(queries), correct)


def tune_margin(index: IntentIndex, vectors, queries: Sequence) -> float:
    """Highest served accuracy at coverage >= MIN_COVERAGE, on the tuning slice.

    Falls back to the widest-coverage point if nothing clears the floor, which is
    what happens when the grid is scaled for a different encoder.
    """
    points = [measure(index, vectors, queries, margin) for margin in MARGIN_GRID]
    eligible = [p for p in points if p.coverage >= MIN_COVERAGE]
    if eligible:
        return max(eligible, key=lambda p: p.served_accuracy).margin
    return max(points, key=lambda p: p.coverage).margin


def _build(model_name: str, canonical_path: Path) -> tuple[IntentIndex, list]:
    examples = load_canonical_examples(canonical_path)
    vectors = encode_texts([e.text for e in examples], model_name=model_name)
    return IntentIndex(
        examples, vectors, model_name, "sha256:operating-point"
    ), examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", type=Path, default=Path("data/intent_canonical.json")
    )
    parser.add_argument(
        "--eval-queries", type=Path, default=Path("data/intent_eval_queries.json")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_SLICE_SIZE)
    args = parser.parse_args(argv)

    split = split_eval_queries(
        load_intent_eval_queries(args.eval_queries),
        slice_size=args.slice_size,
        seed=args.seed,
    )
    print(
        f"split: seed {args.seed}, tuning {len(split.tuning)}, test {len(split.test)}\n"
    )

    tuned: dict[str, OperatingPoint] = {}
    for label, model_name in (
        ("previous", PREVIOUS_ENCODER),
        ("serving", DEFAULT_ENCODER),
    ):
        index, _ = _build(model_name, args.canonical)
        tuning_vectors = encode_texts(
            [q.text for q in split.tuning], model_name=model_name
        )
        test_vectors = encode_texts([q.text for q in split.test], model_name=model_name)
        margin = tune_margin(index, tuning_vectors, split.tuning)
        point = measure(index, test_vectors, split.test, margin)
        tuned[model_name] = point
        print(
            f"{label:>8} {model_name}\n"
            f"         tuned min_margin {margin:.3f} on the tuning slice\n"
            f"         test: coverage {point.coverage:.4f} ({point.served}/{point.total}) "
            f"served accuracy {point.served_accuracy:.4f} "
            f"wrong routes {point.wrong_routes}\n"
        )

    # Matched coverage: the previous encoder's coverage is the reference, because
    # comparing at each model's own point conflates "answers less" with "answers
    # better". Report the serving encoder at whichever margin lands nearest.
    reference = tuned[PREVIOUS_ENCODER].coverage
    index, _ = _build(DEFAULT_ENCODER, args.canonical)
    test_vectors = encode_texts(
        [q.text for q in split.test], model_name=DEFAULT_ENCODER
    )
    points = [measure(index, test_vectors, split.test, m) for m in MARGIN_GRID]
    matched = min(points, key=lambda p: abs(p.coverage - reference))
    print(
        f"matched coverage (reference {reference:.4f} from the previous encoder)\n"
        f"         {DEFAULT_ENCODER} at min_margin {matched.margin:.3f}: "
        f"coverage {matched.coverage:.4f} ({matched.served}/{matched.total}) "
        f"served accuracy {matched.served_accuracy:.4f} "
        f"wrong routes {matched.wrong_routes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
