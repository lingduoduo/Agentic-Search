"""CLI and library for offline retrieval evaluation.

Usage:
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl \\
        --top_k 10

QA pairs file format (one JSON object per line):
    {"query": "...", "relevant_doc_ids": ["doc-id-1", "doc-id-2"]}
"""

from __future__ import annotations

import argparse
import json

from .eval_metrics import mrr as mrr_score
from .eval_metrics import ndcg_at_k, recall_at_k
from .service import RetrievalService


def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
) -> dict[str, float | int]:
    """Load QA pairs, run retrieval, compute and return averaged metrics."""
    _service = service or RetrievalService.from_env()

    with open(dataset_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    recalls, ndcgs, mrrs = [], [], []
    for item in qa_pairs:
        query: str = item["query"]
        relevant: set[str] = set(item["relevant_doc_ids"])
        results, _ = _service.search(query, top_k=top_k)
        retrieved = [r.doc_id for r in results]
        recalls.append(recall_at_k(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        mrrs.append(mrr_score(retrieved, relevant))

    n = len(qa_pairs)
    return {
        f"recall@{top_k}": sum(recalls) / n if n else 0.0,
        f"ndcg@{top_k}": sum(ndcgs) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
        "num_queries": n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument("--dataset", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    metrics = run_eval(args.dataset, top_k=args.top_k)
    print(json.dumps(metrics, indent=2))
