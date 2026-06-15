"""BEIR evaluation script.

Downloads BEIR benchmark datasets (nfcorpus, fiqa, scifact) via the `beir` package,
wraps a RetrievalService, and computes Recall@10, NDCG@10, and MRR per dataset.

Usage:
    python -m src.internal.retrieval.beir_eval \\
        --datasets nfcorpus fiqa scifact \\
        --top_k 10 \\
        [--baseline data/eval/beir_baseline.json]

Set RETRIEVAL_BACKEND (and backend-specific env vars) before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .eval_metrics import ndcg_at_k, recall_at_k, mrr
from .service import RetrievalService

logger = logging.getLogger(__name__)

_SUPPORTED_DATASETS = ("nfcorpus", "fiqa", "scifact", "arguana", "webis-touche2020")


def _load_beir_dataset(dataset_name: str, data_dir: str) -> tuple[dict, dict, dict]:
    """Load BEIR corpus, queries, and qrels via the beir package."""
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise ImportError(
            "BEIR eval requires the 'beir' package. Install with: pip install beir"
        ) from exc

    dataset_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        logger.info("Downloading BEIR dataset %s from %s", dataset_name, url)
        beir_util.download_and_unzip(url, data_dir)

    loader = GenericDataLoader(dataset_path)
    corpus, queries, qrels = loader.load(split="test")
    return corpus, queries, qrels


def _run_beir_dataset(
    dataset_name: str,
    data_dir: str,
    service: RetrievalService,
    top_k: int,
) -> dict[str, float]:
    """Evaluate one BEIR dataset and return metrics dict."""
    corpus, queries, qrels = _load_beir_dataset(dataset_name, data_dir)

    recall_sum = ndcg_sum = mrr_sum = 0.0
    count = 0

    for query_id, query_text in queries.items():
        relevant_ids = {
            doc_id for doc_id, rel in qrels.get(query_id, {}).items() if rel > 0
        }
        if not relevant_ids:
            continue

        try:
            results, _ = service.search(query_text, top_k=top_k)
        except Exception as exc:
            logger.warning("Search failed for query %s: %s", query_id, exc)
            continue

        retrieved = [r.doc_id for r in results]
        recall_sum += recall_at_k(retrieved, relevant_ids, k=top_k)
        ndcg_sum += ndcg_at_k(retrieved, relevant_ids, k=top_k)
        mrr_sum += mrr(retrieved, relevant_ids)
        count += 1

    if count == 0:
        return {"recall@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "num_queries": 0}

    return {
        f"recall@{top_k}": round(recall_sum / count, 4),
        f"ndcg@{top_k}": round(ndcg_sum / count, 4),
        "mrr": round(mrr_sum / count, 4),
        "num_queries": count,
    }


def run_beir_eval(
    datasets: list[str],
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    data_dir: str = "data/beir",
) -> dict[str, dict[str, float]]:
    """Run BEIR eval on the given datasets.

    Returns a dict keyed by dataset name with per-dataset metric dicts.
    """
    svc = service or RetrievalService.from_env()
    results: dict[str, dict[str, float]] = {}
    for name in datasets:
        if name not in _SUPPORTED_DATASETS:
            logger.warning("Unknown BEIR dataset %r — skipping", name)
            continue
        logger.info("Evaluating dataset: %s", name)
        results[name] = _run_beir_dataset(name, data_dir, svc, top_k)
        logger.info("%s: %s", name, results[name])
    return results


def _check_regression(
    current: dict[str, dict[str, float]],
    baseline_path: str,
    *,
    recall_drop_threshold: float = 0.02,
    ndcg_drop_threshold: float = 0.01,
) -> list[str]:
    """Return list of regression messages (empty = no regressions)."""
    baseline_path_obj = Path(baseline_path)
    if not baseline_path_obj.exists():
        logger.info("No baseline file at %s — skipping regression check", baseline_path)
        return []

    baseline: dict = json.loads(baseline_path_obj.read_text())
    failures: list[str] = []

    for dataset, metrics in current.items():
        base = baseline.get(dataset, {})
        for metric_key, threshold in [
            ("recall@10", recall_drop_threshold),
            ("ndcg@10", ndcg_drop_threshold),
        ]:
            base_val = base.get(metric_key)
            curr_val = metrics.get(metric_key)
            if base_val is None or curr_val is None:
                continue
            drop = base_val - curr_val
            if drop > threshold:
                failures.append(
                    f"{dataset}/{metric_key}: dropped {drop:.4f} (baseline={base_val:.4f}, current={curr_val:.4f}, threshold={threshold:.4f})"
                )

    return failures


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="BEIR evaluation for the retrieval service"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus", "fiqa", "scifact"],
        choices=list(_SUPPORTED_DATASETS),
        help="BEIR datasets to evaluate",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--data_dir", default="data/beir", help="Local BEIR data directory"
    )
    parser.add_argument(
        "--output", default=None, help="Write results JSON to this path"
    )
    parser.add_argument(
        "--baseline", default=None, help="Baseline JSON path for regression check"
    )
    args = parser.parse_args()

    results = run_beir_eval(args.datasets, top_k=args.top_k, data_dir=args.data_dir)

    output_str = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output_str)
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)

    if args.baseline:
        failures = _check_regression(results, args.baseline)
        if failures:
            print("\nREGRESSION DETECTED:")
            for f in failures:
                print(f"  {f}")
            raise SystemExit(1)
        else:
            print("\nNo regressions detected.")


if __name__ == "__main__":
    main()
