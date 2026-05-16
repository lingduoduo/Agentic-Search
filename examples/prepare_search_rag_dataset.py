"""Prepare QA + cached retrieval datasets as compact RAG parquet files.

Example:
    python3 -m examples.prepare_search_rag_dataset \
      --dataset_name RUC-NLPIR/FlashRAG_datasets \
      --dataset_config nq \
      --corpus_path data/wiki-18.jsonl \
      --train_retrieval_cache data/nq_train_retrieval_cache.json \
      --test_retrieval_cache data/nq_test_retrieval_cache.json \
      --local_dir data/nq_rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.training.data import make_search_rag_map_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QA rows plus retrieval cache into RAG parquet files."
    )
    parser.add_argument("--dataset_name", default="RUC-NLPIR/FlashRAG_datasets")
    parser.add_argument("--dataset_config", default="nq")
    parser.add_argument("--local_dir", default="./data/nq_rag")
    parser.add_argument("--template_type", default="base")
    parser.add_argument("--data_source", default="nq")
    parser.add_argument("--ability", default="fact-reasoning")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--train_retrieval_cache", required=True)
    parser.add_argument("--test_retrieval_cache", required=True)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional number of rows to convert per split for inspection runs.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print converted question/context/answer pairs and skip parquet writing.",
    )
    parser.add_argument("--preview_rows", type=int, default=5)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to convert.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_corpus_by_id(corpus_path: str) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    with open(corpus_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            doc = json.loads(line)
            corpus[str(doc["id"])] = doc
    return corpus


def convert_split(
    dataset: Any,
    *,
    split: str,
    retrieval_cache: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    split_dataset = dataset[split]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max_examples must be positive when provided.")
        split_dataset = split_dataset.select(
            range(min(args.max_examples, len(split_dataset)))
        )

    process_fn = make_search_rag_map_fn(
        split,
        retrieval_cache=retrieval_cache,
        corpus=corpus,
        topk=args.topk,
        data_source=args.data_source,
        ability=args.ability,
        template_type=args.template_type,
    )
    return split_dataset.map(
        process_fn,
        with_indices=True,
        remove_columns=split_dataset.column_names,
    )


def preview_records(converted: Any, *, split: str, limit: int) -> None:
    print(f"\n[{split}] converted RAG preview")
    for row in converted.select(range(min(limit, len(converted)))):
        prompt = row["prompt"][-1]["content"]
        prompt_before_context = prompt.split("Context:", 1)[0].strip()
        question = prompt_before_context.split("Question:", 1)[-1].strip()
        reward_target = row["reward_model"]["ground_truth"]["target"]
        preview = {
            "question": question,
            "reward_target": reward_target,
            "prompt_roles": [message["role"] for message in row["prompt"]],
            "context_excerpt": prompt.split("Context:", 1)[-1].strip()[:300],
            "extra_info": row["extra_info"],
        }
        print(json.dumps(preview, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.local_dir)
    if not args.preview:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import datasets
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Hugging Face datasets. Reinstall project "
            "requirements so datasets and pyarrow versions match: "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    retrieval_cache = load_json(args.train_retrieval_cache)
    retrieval_cache.update(load_json(args.test_retrieval_cache))
    corpus = load_corpus_by_id(args.corpus_path)

    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    for split in args.splits:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split {split!r} not found. Available: {available}")
        converted = convert_split(
            dataset,
            split=split,
            retrieval_cache=retrieval_cache,
            corpus=corpus,
            args=args,
        )
        if args.preview:
            preview_records(converted, split=split, limit=args.preview_rows)
        else:
            converted.to_parquet(output_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()
