"""CLI to build a FAISS HNSW index from a corpus.jsonl file.

Usage:
    python -m src.internal.retrieval.indexer \
        --corpus data/corpus.jsonl \
        --index  data/indexes/dense/index.faiss \
        --model  intfloat/e5-base-v2

PRD spec: IndexHNSWFlat with ef_construction=128, ef_search=64.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    corpus_path: str = "data/corpus.jsonl"
    index_path: str = "data/indexes/dense/index.faiss"
    model_name: str = "intfloat/e5-base-v2"
    ef_construction: int = 128
    ef_search: int = 64
    hnsw_m: int = 32
    batch_size: int = 256
    extra: dict = field(default_factory=dict)


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _load_corpus(corpus_path: str) -> list[str]:
    texts: list[str] = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            texts.append(doc.get("text") or doc.get("contents") or "")
    return texts


def build_faiss_index(config: IndexerConfig) -> None:
    """Embed corpus and write an IndexHNSWFlat to config.index_path.

    Skips writing if the corpus is empty.
    """
    import faiss

    texts = _load_corpus(config.corpus_path)
    if not texts:
        logger.warning("Empty corpus — no index written to %s", config.index_path)
        return

    embedder = _load_embedder(config.model_name)

    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i : i + config.batch_size]
        vecs = embedder.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        all_vecs.append(vecs.astype(np.float32))

    embeddings = np.vstack(all_vecs)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, config.hnsw_m)
    index.hnsw.efConstruction = config.ef_construction
    index.hnsw.efSearch = config.ef_search
    index.add(embeddings)

    faiss.write_index(index, config.index_path)
    logger.info("Indexed %d vectors → %s", index.ntotal, config.index_path)
    print(f"Indexed {index.ntotal} vectors → {config.index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS HNSW index from corpus.jsonl"
    )
    parser.add_argument(
        "--corpus", default="data/corpus.jsonl", help="Path to corpus.jsonl"
    )
    parser.add_argument(
        "--index", default="data/indexes/dense/index.faiss", help="Output index path"
    )
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    build_faiss_index(
        IndexerConfig(
            corpus_path=args.corpus,
            index_path=args.index,
            model_name=args.model,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            hnsw_m=args.hnsw_m,
            batch_size=args.batch_size,
        )
    )
