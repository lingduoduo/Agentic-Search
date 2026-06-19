"""FAISS IVF-PQ index builder and HNSW ef_search auto-tuner.

IVF-PQ cuts vector memory ~10x vs IndexHNSWFlat with <2pp recall loss.
HNSWTuner finds the minimum ef_search that meets a recall target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .eval_metrics import recall_at_k

logger = logging.getLogger(__name__)


@dataclass
class FaissIndexConfig:
    nlist: int = 4096
    m: int = 96
    nbits: int = 8
    nprobe: int = 64
    index_type: str = "ivfpq"


class FAISSIndexBuilder:
    def __init__(self, config: FaissIndexConfig | None = None) -> None:
        self.config = config or FaissIndexConfig()

    def build_ivfpq(
        self,
        embeddings: "np.ndarray",
        *,
        training_sample: int = 500_000,
    ) -> "object":
        """Build a trained IVF-PQ index from a float32 embeddings matrix."""
        import faiss  # lazy import — requires conda install faiss-cpu

        cfg = self.config
        d = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, cfg.nlist, cfg.m, cfg.nbits)
        index.nprobe = cfg.nprobe

        n_train = min(training_sample, len(embeddings))
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(embeddings), size=n_train, replace=False)
        train_vecs = embeddings[sample_idx].astype("float32")
        logger.info(
            "Training IVF-PQ index on %d vectors (d=%d, nlist=%d, m=%d)",
            n_train,
            d,
            cfg.nlist,
            cfg.m,
        )
        index.train(train_vecs)
        index.add(embeddings.astype("float32"))
        logger.info(
            "IVF-PQ index built: %d vectors, nprobe=%d", index.ntotal, cfg.nprobe
        )
        return index


class HNSWTuner:
    """Finds minimum ef_search that meets a recall target on an HNSW index."""

    def calibrate(
        self,
        index: object,
        qa_pairs: list[dict],
        embedder: Callable[[str], "np.ndarray"],
        *,
        target_recall_at_10: float = 0.80,
        ef_search_candidates: list[int] | None = None,
    ) -> int:
        """Return smallest ef_search that achieves target_recall_at_10."""
        candidates = ef_search_candidates or [16, 32, 64, 96, 128, 192, 256]

        for ef in candidates:
            index.hnsw.efSearch = ef
            recalls: list[float] = []
            for item in qa_pairs:
                q_vec = embedder(item["query"]).reshape(1, -1).astype("float32")
                _, indices = index.search(q_vec, 10)
                retrieved = [str(i) for i in indices[0] if i >= 0]
                relevant = set(item["relevant_doc_ids"])
                recalls.append(recall_at_k(retrieved, relevant, 10))
            avg = sum(recalls) / len(recalls) if recalls else 0.0
            logger.debug("ef_search=%d recall@10=%.4f", ef, avg)
            if avg >= target_recall_at_10:
                logger.info("HNSWTuner: ef_search=%d achieves recall@10=%.4f", ef, avg)
                return ef

        logger.warning(
            "HNSWTuner: target recall %.2f not met; returning max ef_search=%d",
            target_recall_at_10,
            candidates[-1],
        )
        return candidates[-1]
