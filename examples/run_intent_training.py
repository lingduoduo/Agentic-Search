"""Generate intent examples and train the intent classifier.

Usage
-----
# Step 1: generate training examples from the local corpus
python3 -m examples.run_intent_training generate \
    --corpus data/corpus.jsonl \
    --vocabulary data/vocabulary_corpus.json \
    --output data/intent_examples.json

# Step 2: train and save the classifier
python3 -m examples.run_intent_training train \
    --examples data/intent_examples.json \
    --output models/intent_classifier.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.model.intent_training import (
    generate_intent_examples,
    train_intent_classifier,
    write_intent_examples,
)


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate",
        help="Generate intent-training examples from a corpus and vocabulary.",
    )
    parser.add_argument(
        "--corpus", default="data/corpus.jsonl", help="Path to corpus JSONL"
    )
    parser.add_argument(
        "--vocabulary",
        default="data/vocabulary_corpus.json",
        help="Path to vocabulary metadata JSON",
    )
    parser.add_argument(
        "--output",
        default="data/intent_examples.sample.json",
        help="Path to output JSON examples",
    )
    parser.set_defaults(func=_run_generate)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train and save the intent classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--examples",
        required=True,
        help="JSON file of intent-labeled training examples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the trained pipeline (.pt file)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--min_freq", type=int, default=2, help="Minimum token frequency for vocabulary"
    )
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.set_defaults(func=_run_train)


def _run_generate(args: argparse.Namespace) -> None:
    examples = generate_intent_examples(
        corpus_path=Path(args.corpus),
        vocabulary_path=Path(args.vocabulary),
    )
    output_path = Path(args.output)
    write_intent_examples(examples, output_path)
    print(f"Wrote {len(examples)} examples to {output_path}")


def _run_train(args: argparse.Namespace) -> None:
    print(f"Loading training examples from {args.examples}")
    print(f"Training for {args.epochs} epoch(s)")
    t0 = time.perf_counter()
    try:
        result = train_intent_classifier(
            examples_path=Path(args.examples),
            output_path=Path(args.output),
            epochs=args.epochs,
            lr=args.lr,
            min_freq=args.min_freq,
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        )
        pipeline = result.pipeline
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except ModuleNotFoundError as exc:
        print(
            f"Missing Python dependency {exc.name!r}. Install the project "
            "requirements first, for example: "
            "`python3 -m pip install -r requirements.txt`."
        )
        raise SystemExit(1) from exc

    elapsed = time.perf_counter() - t0
    print(f"  {result.num_examples} examples  |  labels: {result.label_counts}")
    print(f"  training complete in {elapsed:.1f}s")
    print(f"  vocabulary size: {len(pipeline._vocab.token2idx)} tokens")
    print(f"Saved -> {Path(args.output)}")

    samples = [
        ("buy a noise-cancelling headphone", "purchase"),
        ("what is FAISS?", "qa"),
        ("show me the documentation page", "navigate"),
        ("recommend a good search library", "recommendation"),
    ]
    print("\nSmoke test:")
    for text, expected in samples:
        pred = pipeline.predict_text(text)
        match = "ok" if pred.intent == expected else "mismatch"
        print(
            f"  {match:8s} {text!r:45s} -> {pred.intent} "
            f"({pred.confidence:.2f}) [expected: {expected}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate intent examples or train the intent classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(subparsers)
    _add_train_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
