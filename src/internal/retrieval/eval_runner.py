"""CLI and library for offline retrieval evaluation.

Usage (retrieval only — requires BM25_INDEX_PATH env var):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10

Usage against a running retrieval server (no env vars needed):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10 \\
        --retrieval_url http://localhost:8001/retrieve

Usage (with reranking):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10 \\
        --retrieval_url http://localhost:8001/retrieve \\
        --reranker local --reranker_model BAAI/bge-reranker-v2-m3

QA pairs file format (one JSON object per line):
    {"query": "...", "relevant_doc_ids": ["doc-id-1", "doc-id-2"]}
"""

from __future__ import annotations

import argparse
import json
import math
import time

from .eval_metrics import mrr as mrr_score
from .eval_metrics import ndcg_at_k, recall_at_k
from .service import RetrievalService


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(len(sorted_vals) * p / 100) - 1))
    return round(sorted_vals[idx], 1)


def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    reranker=None,  # Reranker | None — avoid circular import at module level
) -> dict:
    """Load QA pairs, run retrieval (and optionally reranking), return metrics.

    Without reranker: returns flat dict {recall@k, ndcg@k, mrr, num_queries}.
    With reranker:    returns {retrieval: {...}, reranked: {...}, latency_ms: {...}}.
    """
    _service = service or RetrievalService.from_env()

    with open(dataset_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    recalls, ndcgs, mrrs = [], [], []
    r_recalls, r_ndcgs, r_mrrs, latencies_ms = [], [], [], []

    for item in qa_pairs:
        query: str = item["query"]
        relevant: set[str] = set(item["relevant_doc_ids"])
        results, _ = _service.search(query, top_k=top_k)
        retrieved = [r.doc_id for r in results]

        recalls.append(recall_at_k(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        mrrs.append(mrr_score(retrieved, relevant))

        if reranker is not None:
            t0 = time.monotonic()
            reranked_results = reranker.rerank(query, results, top_k)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            r_retrieved = [r.doc_id for r in reranked_results]
            r_recalls.append(recall_at_k(r_retrieved, relevant, top_k))
            r_ndcgs.append(ndcg_at_k(r_retrieved, relevant, top_k))
            r_mrrs.append(mrr_score(r_retrieved, relevant))

    n = len(qa_pairs)

    def _avg(lst):
        return round(sum(lst) / n, 4) if n else 0.0

    retrieval_metrics = {
        f"recall@{top_k}": _avg(recalls),
        f"ndcg@{top_k}": _avg(ndcgs),
        "mrr": _avg(mrrs),
        "num_queries": n,
    }

    if reranker is None:
        return retrieval_metrics

    return {
        "retrieval": retrieval_metrics,
        "reranked": {
            f"recall@{top_k}": _avg(r_recalls),
            f"ndcg@{top_k}": _avg(r_ndcgs),
            "mrr": _avg(r_mrrs),
            "num_queries": n,
        },
        "latency_ms": {
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "p99": _percentile(latencies_ms, 99),
            "n": n,
        },
    }


class _HttpService:
    """Thin sync wrapper around the retrieval HTTP endpoint for CLI use."""

    def __init__(self, url: str) -> None:
        self._url = url

    def search(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> tuple[list, str]:
        import requests

        from src.internal.retrieval.backends.base import RetrievalResult

        payload: dict = {"queries": [query], "topk": top_k, "return_scores": True}
        if filters:
            payload["filters"] = filters
        resp = requests.post(self._url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("result", [[]])[0]
        results = []
        for item in rows:
            if "document" in item:
                doc, score = item["document"], float(item.get("score", 0.0))
            else:
                doc, score = item, 0.0
            results.append(
                RetrievalResult(
                    doc_id=str(doc.get("id") or doc.get("doc_id", "")),
                    title=doc.get("title") or "",
                    text=doc.get("contents") or doc.get("text") or "",
                    url=doc.get("url"),
                    score=score,
                )
            )
        return results, "http"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument("--dataset", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--retrieval_url",
        default=None,
        help="HTTP retrieval endpoint (e.g. http://localhost:8001/retrieve). "
        "When omitted, RetrievalService.from_env() is used (requires BM25_INDEX_PATH).",
    )
    parser.add_argument(
        "--reranker",
        choices=["local", "cohere"],
        default=None,
        help="Provider for reranking (omit to skip reranking)",
    )
    parser.add_argument(
        "--reranker_model",
        default="BAAI/bge-reranker-v2-m3",
        help="Model name for local reranker or Cohere model name",
    )
    args = parser.parse_args()

    service = _HttpService(args.retrieval_url) if args.retrieval_url else None

    reranker = None
    if args.reranker:
        import os

        from src.internal.retrieval.reranker import Reranker, RerankerConfig

        reranker = Reranker(
            RerankerConfig(
                provider=args.reranker,
                model=args.reranker_model,
                api_key=os.environ.get("COHERE_API_KEY"),
            )
        )

    metrics = run_eval(
        args.dataset, top_k=args.top_k, service=service, reranker=reranker
    )
    print(json.dumps(metrics, indent=2))
