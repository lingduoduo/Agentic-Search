"""Prepare QA datasets as search-agent prompt/answer parquet files.

Example:
    python3 -m examples.prepare_search_qa_dataset \
      --dataset_name RUC-NLPIR/FlashRAG_datasets \
      --dataset_config nq \
      --local_dir data/nq_search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.training.data import make_search_qa_map_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QA rows into search-agent prompt parquet files."
    )
    parser.add_argument(
        "--dataset_name",
        default="RUC-NLPIR/FlashRAG_datasets",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset_config",
        default="nq",
        help="Optional Hugging Face dataset config/subset.",
    )
    parser.add_argument("--local_dir", default="./data/nq_search")
    parser.add_argument("--template_type", default="base")
    parser.add_argument("--data_source", default="nq")
    parser.add_argument("--ability", default="fact-reasoning")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional number of rows to convert per split for inspection runs.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print converted question/answer pairs and skip parquet writing.",
    )
    parser.add_argument(
        "--preview_rows",
        type=int,
        default=5,
        help="Number of converted rows to print when --preview is set.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to convert.",
    )
    return parser.parse_args()


def convert_split(dataset: Any, *, split: str, args: argparse.Namespace) -> Any:
    split_dataset = dataset[split]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max_examples must be positive when provided.")
        split_dataset = split_dataset.select(
            range(min(args.max_examples, len(split_dataset)))
        )

    process_fn = make_search_qa_map_fn(
        split,
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
    print(f"\n[{split}] converted preview")
    for row in converted.select(range(min(limit, len(converted)))):
        reward_target = row["reward_model"]["ground_truth"]["target"]
        preview = {
            "question": row["prompt"][-1]["content"],
            "golden_answers": reward_target,
            "reward_target": reward_target,
            "prompt_roles": [message["role"] for message in row["prompt"]],
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
            "Failed to import Hugging Face datasets. If you see a pyarrow "
            "extension error, reinstall project requirements so the pyarrow "
            "version matches datasets, for example: "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    for split in args.splits:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split {split!r} not found. Available: {available}")
        converted = convert_split(dataset, split=split, args=args)
        if args.preview:
            preview_records(converted, split=split, limit=args.preview_rows)
        else:
            converted.to_parquet(output_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()
