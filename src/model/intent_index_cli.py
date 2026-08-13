"""Build and evaluate the canonical routing index.

Three commands. ``seed`` proposes module labels for the existing training
examples so the canonical set can be curated rather than authored from nothing.
``build`` encodes the canonical set into an index. ``evaluate`` scores an index
against the held-out query sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .intent_data import load_canonical_examples
from .intent_encoder import DEFAULT_ENCODER, encode_texts
from .intent_knn import INDEX_FILENAME, IntentIndex

LEAKAGE_COSINE = 0.95


def _fingerprint(path: Path) -> str:
    """Hash the canonical file so a stale index is detectable."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def build_index(
    canonical_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_ENCODER,
    encode=None,
) -> IntentIndex:
    """Encode the canonical examples and write the index.

    ``encode`` is injectable so the build is testable without an encoder; it is
    not a production knob. It defaults to None rather than to ``encode_texts``
    so the module attribute is resolved at call time — a default argument binds
    at definition, which would put the real encoder beyond reach of a test's
    monkeypatch.
    """
    encoder = encode if encode is not None else encode_texts
    examples = load_canonical_examples(canonical_path)
    vectors = encoder([example.text for example in examples], model_name=model_name)
    index = IntentIndex(
        examples=examples,
        vectors=vectors,
        encoder=model_name,
        fingerprint=_fingerprint(canonical_path),
    )
    index.save(output_dir / INDEX_FILENAME)
    return index


def check_leakage(
    index: IntentIndex, eval_texts: Sequence[str], eval_vectors: np.ndarray
) -> list[str]:
    """Report evaluation queries that duplicate a canonical example.

    With nearest-neighbor routing the index *is* the model, so an eval query
    that also sits in the index scores against itself and manufactures accuracy
    that no user would ever see.
    """
    canonical_texts = [example.text for example in index.examples]
    normalized = {text.casefold().strip(): text for text in canonical_texts}
    similarities = eval_vectors @ index.vectors.T
    leaks: list[str] = []
    for position, text in enumerate(eval_texts):
        exact = normalized.get(text.casefold().strip())
        if exact is not None:
            leaks.append(f"{text!r} exactly matches canonical {exact!r}")
            continue
        best = int(np.argmax(similarities[position]))
        score = float(similarities[position][best])
        if score >= LEAKAGE_COSINE:
            leaks.append(
                f"{text!r} is {score:.3f} similar to canonical "
                f"{canonical_texts[best]!r}"
            )
    return leaks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser(
        "seed", help="propose module labels for existing training examples"
    )
    seed.add_argument("--examples", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)

    build = subparsers.add_parser("build", help="encode the canonical set")
    build.add_argument("--canonical", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--model", default=DEFAULT_ENCODER)

    evaluate = subparsers.add_parser("evaluate", help="score an index")
    evaluate.add_argument("--index", required=True, type=Path)
    evaluate.add_argument("--eval-queries", required=True, type=Path)
    evaluate.add_argument("--hard-queries", type=Path)
    evaluate.add_argument("--out-of-scope", type=Path)
    evaluate.add_argument("--canonical", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a seed, build, or evaluate command."""
    try:
        args = _build_parser().parse_args(argv)
        if args.command == "seed":
            from .intent_seed import write_seed_canonical

            count = write_seed_canonical(args.examples, args.output)
            print(f"wrote {count} proposed canonical examples to {args.output}")
            return 0
        if args.command == "build":
            index = build_index(args.canonical, args.output, model_name=args.model)
            print(f"built index of {index.size} examples at {args.output}")
            low = index.low_support_modules()
            print(f"low support modules ({len(low)}): {', '.join(low) or 'none'}")
            return 0

        from .intent_index_eval import run_index_evaluation

        report = run_index_evaluation(
            index_path=args.index,
            eval_queries_path=args.eval_queries,
            hard_queries_path=args.hard_queries,
            out_of_scope_path=args.out_of_scope,
            canonical_path=args.canonical,
            output_path=args.output,
        )
        print(json.dumps(report["headline"], indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"intent index command failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
