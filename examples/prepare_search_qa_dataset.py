"""Prepare QA datasets as search-agent prompt/answer parquet files.

Example:
    python3 -m examples.prepare_search_qa_dataset \
      --dataset_name RUC-NLPIR/FlashRAG_datasets \
      --dataset_config nq \
      --local_dir data/nq_search
"""

from __future__ import annotations

import argparse
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
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to convert.",
    )
    return parser.parse_args()


def convert_split(dataset: Any, *, split: str, args: argparse.Namespace) -> Any:
    split_dataset = dataset[split]
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


def main() -> None:
    import datasets

    args = parse_args()
    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    for split in args.splits:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split {split!r} not found. Available: {available}")
        converted = convert_split(dataset, split=split, args=args)
        converted.to_parquet(output_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()
