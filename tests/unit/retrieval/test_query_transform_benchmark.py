from __future__ import annotations

from src.context.query_transform import QueryTransformConfig
from src.internal.retrieval.query_transform_benchmark import (
    run_query_transform_benchmark,
)


def test_benchmark_ranks_configs():
    dataset = [("q1", {"d1"}), ("q2", {"d2"})]

    def retrieve(query, config):
        # A config that decomposes "finds" the right doc; the other does not.
        if config.decompose:
            return {"q1": ["d1"], "q2": ["d2"]}[query]
        return ["dx"]

    configs = [QueryTransformConfig(), QueryTransformConfig(decompose=True)]
    rows = run_query_transform_benchmark(dataset, retrieve, configs, k=5)
    best = max(rows, key=lambda r: r["recall"])
    assert best["recall"] == 1.0
    assert "mean_latency_ms" in best and "config_signature" in best
