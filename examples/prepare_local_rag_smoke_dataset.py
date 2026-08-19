from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.internal.servers.retrieval.demo import TfidfRetriever
from src.model.post_training.data import build_search_rag_record, format_rag_reference


SMOKE_EXAMPLES: tuple[dict[str, object], ...] = (
    {
        "question": "What system enables efficient nearest-neighbor search over millions of high-dimensional vectors?",
        "golden_answers": ["FAISS"],
    },
    {
        "question": "What ranking function uses term frequency and inverse document frequency?",
        "golden_answers": ["BM25"],
    },
    {
        "question": "What does retrieval-augmented generation combine?",
        "golden_answers": ["a retriever and a generative language model"],
    },
    {
        "question": "Which Python web framework provides automatic OpenAPI documentation?",
        "golden_answers": ["FastAPI"],
    },
)


def _validate_corpus_path(corpus_path: str | Path) -> Path:
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {path}")
    return path


def _context_documents(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": hit["document"]["id"],
            "title": hit["document"].get("title", ""),
            "contents": hit["document"]["text"],
        }
        for hit in hits
    ]


def build_smoke_records(
    corpus_path: str | Path,
    *,
    topk: int = 3,
) -> list[dict[str, object]]:
    if topk < 1:
        raise ValueError("topk must be at least 1.")
    path = _validate_corpus_path(corpus_path)
    try:
        retriever = TfidfRetriever(str(path))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed corpus JSONL at {path}:{exc.lineno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError(f"Corpus has no searchable text: {path}") from exc
        raise
    questions = [str(example["question"]) for example in SMOKE_EXAMPLES]
    retrieval_rows = retriever.retrieve(questions, topk)

    records: list[dict[str, object]] = []
    for index, (example, hits) in enumerate(zip(SMOKE_EXAMPLES, retrieval_rows)):
        if not hits:
            raise ValueError(
                f"No retrieval results for smoke question: {example['question']}"
            )
        context = format_rag_reference(_context_documents(hits))
        records.append(
            build_search_rag_record(
                example,
                context=context,
                split="smoke",
                index=index,
                data_source="local-demo",
                ability="fact-reasoning",
            )
        )
    return records


def preview_records(records: list[dict[str, object]]) -> None:
    print("Local RAG smoke-test preview")
    for record in records:
        prompt = record["prompt"][0]["content"]
        context = prompt.split("Context:\n", 1)[-1]
        preview = {
            "question": prompt.split("Question:", 1)[-1]
            .split("Context:", 1)[0]
            .strip(),
            "reward_target": record["reward_model"]["ground_truth"]["target"],
            "context_excerpt": context[:300],
            "extra_info": record["extra_info"],
        }
        print(json.dumps(preview, ensure_ascii=False))


def write_parquet(records: list[dict[str, object]], output_path: str | Path) -> Path:
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Hugging Face datasets; install project requirements."
        ) from exc
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small offline RAG parquet dataset from the demo corpus."
    )
    parser.add_argument("--corpus_path", default="data/corpus.jsonl")
    parser.add_argument("--output_path", default="data/local_rag_smoke.parquet")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print converted records and skip parquet writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_smoke_records(args.corpus_path, topk=args.topk)
    if args.preview:
        preview_records(records)
        return
    output_path = write_parquet(records, args.output_path)
    print(f"Wrote {len(records)} local RAG smoke records to {output_path}")


if __name__ == "__main__":
    main()
