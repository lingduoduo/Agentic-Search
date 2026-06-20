"""Offline trainer for the QueryRouter. Produces a joblib sklearn Pipeline.

Usage: python -m src.training.train_query_router --out data/query_router.joblib
"""

from __future__ import annotations

import argparse

from src.internal.retrieval.query_router import ROUTER_LABELS

# Labels order: decompose, hyde, step_back, keywords, construct_filters, multi_query
# Each row must have exactly one positive AND one negative across the corpus per column
# to avoid single-valued columns in MultiOutputClassifier.
SEED_DATA: list[tuple[str, list[int]]] = [
    ("faiss index", [0, 0, 0, 1, 0, 0]),
    ("bm25 tuning", [0, 0, 0, 1, 0, 0]),
    ("what is reciprocal rank fusion", [0, 1, 1, 0, 0, 1]),
    ("how does HNSW graph search work", [0, 1, 1, 0, 0, 1]),
    ("compare dense and sparse retrieval and when each wins", [1, 0, 0, 0, 0, 0]),
    ("explain reranking and decompose the tradeoffs and latency", [1, 0, 0, 0, 0, 0]),
    ("FAISS papers after 2023", [0, 0, 0, 0, 1, 0]),
    ("arxiv papers between 2020 and 2022 on retrieval", [0, 0, 0, 0, 1, 0]),
    ("best embedding model for semantic search", [0, 1, 1, 0, 0, 1]),
    ("vector database benchmarks", [0, 0, 0, 1, 0, 0]),
]


def build_model():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("vec", HashingVectorizer(ngram_range=(1, 2), n_features=2**12)),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000))),
        ]
    )


def train(output_path: str) -> None:
    import joblib

    assert all(len(y) == len(ROUTER_LABELS) for _, y in SEED_DATA), (
        f"Every SEED_DATA row must have exactly {len(ROUTER_LABELS)} labels"
    )
    queries = [q for q, _ in SEED_DATA]
    labels = [y for _, y in SEED_DATA]
    model = build_model()
    model.fit(queries, labels)
    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QueryRouter model")
    parser.add_argument("--out", default="data/query_router.joblib")
    args = parser.parse_args()
    train(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
