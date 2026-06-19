"""BM25 parameter grid search: finds optimal (k1, b) for a QA pairs dataset.

Usage:
    python -m src.internal.retrieval.bm25_tuner \
        --qa_pairs data/eval/qa_pairs.jsonl \
        --output data/eval/bm25_params.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Callable

from .eval_metrics import recall_at_k

logger = logging.getLogger(__name__)


@dataclass
class BM25Params:
    k1: float
    b: float
    score: float


class BM25Tuner:
    """Grid-searches BM25 k1/b parameters against labeled QA pairs.

    Args:
        service_factory: Callable[(k1, b) -> RetrievalService]. Called once
            per grid point so the backend can be rebuilt with new params.
    """

    def __init__(self, service_factory: Callable[[float, float], object]) -> None:
        self._factory = service_factory

    def grid_search(
        self,
        qa_pairs_path: str,
        *,
        k1_range: list[float] | None = None,
        b_range: list[float] | None = None,
        metric: str = "recall@10",
        top_k: int = 10,
        output_path: str | None = None,
    ) -> BM25Params:
        """Return (k1, b) that maximise metric on the QA pairs dataset."""
        k1_range = k1_range or [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        b_range = b_range or [0.3, 0.5, 0.6, 0.75, 0.9]

        with open(qa_pairs_path) as f:
            qa_pairs = [json.loads(line) for line in f if line.strip()]

        best = BM25Params(k1=k1_range[0], b=b_range[0], score=-1.0)

        for k1 in k1_range:
            for b in b_range:
                service = self._factory(k1, b)
                scores: list[float] = []
                for item in qa_pairs:
                    results, _ = service.search(item["query"], top_k=top_k)
                    retrieved = [r.doc_id for r in results]
                    relevant = set(item["relevant_doc_ids"])
                    scores.append(recall_at_k(retrieved, relevant, top_k))
                avg = sum(scores) / len(scores) if scores else 0.0
                logger.debug("k1=%.2f b=%.2f %s=%.4f", k1, b, metric, avg)
                if avg > best.score:
                    best = BM25Params(k1=k1, b=b, score=avg)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(asdict(best), f, indent=2)
            logger.info("BM25 params saved to %s", output_path)

        return best
