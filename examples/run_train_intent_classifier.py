"""Train and save the intent classifier as a standalone offline step.

Usage
-----
# Step 1 (once): generate training examples from the local corpus
python3 -m examples.run_generate_intent_examples \
    --corpus data/corpus.jsonl \
    --vocabulary data/vocabulary_corpus.json \
    --output data/intent_examples.json

# Step 2 (once): train and save the classifier
python3 -m examples.run_train_intent_classifier \
    --examples data/intent_examples.json \
    --output models/intent_classifier.pt

# Step 3 (every run): load the saved model — no retraining
python3 -m examples.run_agentic_search \
    --mode search \
    --question "Buy me a noise-cancelling headphone" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve \
    --intent_model models/intent_classifier.pt

Separating training from inference means the classifier is trained once
on a curated dataset and reused across many agent runs without the startup
cost of re-fitting the model on every invocation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.agent_loop.intent_training import train_intent_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save the intent classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--examples",
        required=True,
        help="JSON file of intent-labelled training examples (from run_generate_intent_examples.py)",
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
    args = parser.parse_args()

    print(f"Loading training examples from {args.examples} …")

    print(f"Training for {args.epochs} epoch(s) …")
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

    print(f"Saved → {Path(args.output)}")

    # Quick smoke test
    samples = [
        ("buy a noise-cancelling headphone", "purchase"),
        ("what is FAISS?", "qa"),
        ("show me the documentation page", "navigate"),
        ("recommend a good search library", "recommendation"),
    ]
    print("\nSmoke test:")
    for text, expected in samples:
        pred = pipeline.predict_text(text)
        match = "✓" if pred.intent == expected else "✗"
        print(
            f"  {match}  {text!r:45s}  → {pred.intent} ({pred.confidence:.2f})  [expected: {expected}]"
        )


if __name__ == "__main__":
    main()
