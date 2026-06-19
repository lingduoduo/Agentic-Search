from __future__ import annotations

import json
import math
import time

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.eval_metrics import map_at_k
from src.internal.retrieval.eval_metrics import mrr as mrr_score
from src.internal.retrieval.eval_metrics import ndcg_at_k
from src.internal.retrieval.passage_truncator import PassageTruncator
from src.internal.retrieval.reranker import Reranker, RerankerConfig


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(len(sorted_vals) * p / 100) - 1))
    return round(sorted_vals[idx], 1)


def run_benchmark(
    qa_pairs_path: str,
    *,
    models: list[str],
    batch_sizes: list[int],
    max_tokens_list: list[int],
    top_k: int = 10,
    output_path: str | None = None,
) -> list[dict]:
    """Grid search over model × batch_size × max_tokens. QA pairs must include 'candidates'."""
    with open(qa_pairs_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    rows = []
    for model in models:
        for batch_size in batch_sizes:
            for max_tokens in max_tokens_list:
                config = RerankerConfig(
                    provider="local", model=model, batch_size=batch_size
                )
                reranker = Reranker(config)
                truncator = PassageTruncator(max_tokens=max_tokens)

                ndcgs, mrrs, maps, latencies = [], [], [], []
                for item in qa_pairs:
                    query: str = item["query"]
                    relevant: set[str] = set(item["relevant_doc_ids"])
                    raw_candidates = item.get("candidates", [])
                    candidates = [RetrievalResult(**c) for c in raw_candidates]
                    # Apply truncation before reranking
                    truncated = [
                        RetrievalResult(
                            doc_id=c.doc_id,
                            title=c.title,
                            text=truncator.truncate(c.text),
                            url=c.url,
                            score=c.score,
                        )
                        for c in candidates
                    ]

                    t0 = time.monotonic()
                    reranked = reranker.rerank(query, truncated, top_k)
                    latencies.append((time.monotonic() - t0) * 1000)

                    retrieved = [r.doc_id for r in reranked]
                    ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
                    mrrs.append(mrr_score(retrieved, relevant))
                    maps.append(map_at_k(retrieved, relevant, top_k))

                n = len(qa_pairs)

                def _avg(lst: list[float]) -> float:
                    return round(sum(lst) / n, 4) if n else 0.0

                row = {
                    "model": model,
                    "batch_size": batch_size,
                    "max_tokens": max_tokens,
                    f"ndcg@{top_k}": _avg(ndcgs),
                    "mrr": _avg(mrrs),
                    f"map@{top_k}": _avg(maps),
                    "mean_ms": round(sum(latencies) / n, 1) if n else 0.0,
                    "p50_ms": _percentile(latencies, 50),
                    "p99_ms": _percentile(latencies, 99),
                }
                rows.append(row)

    if output_path:
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # Print ranked table
    sorted_rows = sorted(rows, key=lambda r: r.get(f"ndcg@{top_k}", 0), reverse=True)
    header = f"{'model':<35} {'batch':>6} {'tok':>5} {f'ndcg@{top_k}':>8} {'mrr':>6} {'p99ms':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted_rows:
        print(
            f"{r['model']:<35} {r['batch_size']:>6} {r['max_tokens']:>5} "
            f"{r[f'ndcg@{top_k}']:>8.4f} {r['mrr']:>6.4f} {r['p99_ms']:>7.1f}"
        )

    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reranker model/config benchmark")
    parser.add_argument(
        "--qa-pairs",
        required=True,
        help="Path to qa_pairs.jsonl (with 'candidates' field)",
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--max-tokens", type=int, nargs="+", default=[512], dest="max_tokens"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_benchmark(
        args.qa_pairs,
        models=args.models,
        batch_sizes=args.batch_sizes,
        max_tokens_list=args.max_tokens,
        top_k=args.top_k,
        output_path=args.output,
    )
