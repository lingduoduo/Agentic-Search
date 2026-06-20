"""Offline grid benchmark over query-transform technique combinations.

Usage: python -m src.internal.retrieval.query_transform_benchmark --dataset path.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

from src.context.query_transform import QueryTransformConfig, config_signature
from src.internal.retrieval.eval_metrics import ndcg_at_k, recall_at_k


def run_query_transform_benchmark(
    dataset: list[tuple[str, set[str]]],
    retrieve_fn,
    configs: list[QueryTransformConfig],
    *,
    k: int = 10,
) -> list[dict]:
    """Evaluate each config against the labeled dataset.

    Args:
        dataset: List of (query, relevant_doc_ids) pairs.
        retrieve_fn: Callable(query, config) -> list[str] of ranked doc ids.
        configs: Technique-combination configs to benchmark.
        k: Cut-off for recall@k and NDCG@k.

    Returns:
        List of result dicts sorted by recall descending, each with keys:
        ``config_signature``, ``recall``, ``ndcg``, ``mean_latency_ms``.
    """
    rows: list[dict] = []
    for config in configs:
        recalls, ndcgs, latencies = [], [], []
        for query, relevant in dataset:
            start = time.perf_counter()
            ranked = retrieve_fn(query, config)
            latencies.append((time.perf_counter() - start) * 1000)
            recalls.append(recall_at_k(ranked, relevant, k))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        n = len(dataset) or 1
        rows.append(
            {
                "config_signature": config_signature(config),
                "recall": sum(recalls) / n,
                "ndcg": sum(ndcgs) / n,
                "mean_latency_ms": sum(latencies) / n,
            }
        )
    return sorted(rows, key=lambda r: r["recall"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark query-transform configs against a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL file with {query, relevant_ids} objects.",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall/NDCG cut-off.")
    args = parser.parse_args()

    dataset: list[tuple[str, set[str]]] = []
    with open(args.dataset) as fh:
        for line in fh:
            obj = json.loads(line)
            dataset.append((obj["query"], set(obj["relevant_ids"])))

    configs = [
        QueryTransformConfig(),
        QueryTransformConfig(decompose=True),
    ]

    # Stub retrieve_fn — replace with a real one when integrating.
    def _stub_retrieve(query: str, config: QueryTransformConfig) -> list[str]:
        raise NotImplementedError(
            "Provide a real retrieve_fn when calling run_query_transform_benchmark directly."
        )

    rows = run_query_transform_benchmark(dataset, _stub_retrieve, configs, k=args.k)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
