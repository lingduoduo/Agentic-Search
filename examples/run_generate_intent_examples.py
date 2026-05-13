"""Generate intent-classification training examples from the local corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agent_loop.intent_training import (
    generate_intent_examples,
    write_intent_examples,
)

generate_examples = generate_intent_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample intent-training data."
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
    args = parser.parse_args()

    examples = generate_intent_examples(
        corpus_path=Path(args.corpus),
        vocabulary_path=Path(args.vocabulary),
    )
    output_path = Path(args.output)
    write_intent_examples(examples, output_path)
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
