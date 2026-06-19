from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.reranker_benchmark import run_benchmark


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _write_qa(path, *, with_candidates=True):
    entry = {
        "query": "test query",
        "relevant_doc_ids": ["d1"],
    }
    if with_candidates:
        entry["candidates"] = [
            {"doc_id": "d1", "title": "t", "text": "c", "url": None, "score": 0.9},
            {"doc_id": "d2", "title": "t", "text": "c", "url": None, "score": 0.5},
        ]
    path.write_text(json.dumps(entry) + "\n")


def test_run_benchmark_returns_results(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("d1"), _result("d2")]

    with patch(
        "src.internal.retrieval.reranker_benchmark.Reranker",
        return_value=fake_reranker,
    ):
        results = run_benchmark(
            str(qa),
            models=["BAAI/bge-reranker-base"],
            batch_sizes=[8],
            max_tokens_list=[256],
        )

    assert len(results) == 1
    row = results[0]
    assert "model" in row
    assert "batch_size" in row
    assert "max_tokens" in row
    assert "ndcg@10" in row
    assert "mrr" in row
    assert "p99_ms" in row
    assert "mean_ms" in row


def test_run_benchmark_output_jsonl(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    out = tmp_path / "bench.jsonl"

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("d1")]

    with patch(
        "src.internal.retrieval.reranker_benchmark.Reranker", return_value=fake_reranker
    ):
        run_benchmark(
            str(qa),
            models=["m1"],
            batch_sizes=[4],
            max_tokens_list=[128],
            output_path=str(out),
        )

    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert len(rows) == 1
