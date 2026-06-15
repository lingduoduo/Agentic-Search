"""Tests for RAGAS evaluation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.ragas_eval import (
    RagasSample,
    _check_regression,
    build_ragas_dataset,
    run_ragas_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(*texts: str) -> MagicMock:
    """Return a mock RetrievalService that returns the given texts as results."""
    results = [
        RetrievalResult(
            doc_id=f"doc-{i}",
            title=f"Title {i}",
            text=t,
            url=None,
            score=1.0 - i * 0.1,
        )
        for i, t in enumerate(texts)
    ]
    svc = MagicMock()
    svc.search.return_value = (results, "hybrid")
    return svc


# ---------------------------------------------------------------------------
# build_ragas_dataset
# ---------------------------------------------------------------------------


def test_build_ragas_dataset_returns_one_sample_per_pair():
    svc = _make_service("FAISS is a library. It supports dense search.")
    qa_pairs = [{"question": "What is FAISS?", "ground_truth": "FAISS is a library."}]

    samples = build_ragas_dataset(qa_pairs, service=svc, top_k=5)

    assert len(samples) == 1
    assert samples[0].question == "What is FAISS?"
    assert samples[0].ground_truth == "FAISS is a library."


def test_build_ragas_dataset_fills_contexts_from_retrieval():
    svc = _make_service("Context A.", "Context B.")
    qa_pairs = [{"question": "Q?", "ground_truth": "G."}]

    samples = build_ragas_dataset(qa_pairs, service=svc, top_k=2)

    assert samples[0].contexts == ["Context A.", "Context B."]


def test_build_ragas_dataset_extractive_answer_uses_first_sentence():
    svc = _make_service("First sentence. Second sentence.")
    qa_pairs = [{"question": "Q?", "ground_truth": "G."}]

    samples = build_ragas_dataset(qa_pairs, service=svc, top_k=1)

    assert samples[0].answer == "First sentence."


def test_build_ragas_dataset_skips_on_retrieval_failure():
    svc = MagicMock()
    svc.search.side_effect = RuntimeError("backend down")
    qa_pairs = [{"question": "Q?", "ground_truth": "G."}]

    samples = build_ragas_dataset(qa_pairs, service=svc)

    assert samples == []


def test_build_ragas_dataset_handles_missing_ground_truth():
    svc = _make_service("Some context.")
    qa_pairs = [{"question": "Q?"}]

    samples = build_ragas_dataset(qa_pairs, service=svc)

    assert samples[0].ground_truth == ""


def test_build_ragas_dataset_empty_contexts_gives_empty_answer():
    svc = MagicMock()
    svc.search.return_value = ([], "sparse_only")
    qa_pairs = [{"question": "Q?", "ground_truth": "G."}]

    samples = build_ragas_dataset(qa_pairs, service=svc)

    assert samples[0].answer == ""
    assert samples[0].contexts == []


# ---------------------------------------------------------------------------
# _check_regression
# ---------------------------------------------------------------------------


def test_check_regression_no_baseline_file_returns_empty(tmp_path: Path):
    result = _check_regression({"faithfulness": 0.8}, str(tmp_path / "missing.json"))
    assert result == []


def test_check_regression_no_regression_within_threshold(tmp_path: Path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"faithfulness": 0.80, "answer_relevancy": 0.75}))

    current = {"faithfulness": 0.76, "answer_relevancy": 0.71}
    failures = _check_regression(current, str(baseline))

    assert failures == []


def test_check_regression_detects_faithfulness_drop(tmp_path: Path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"faithfulness": 0.80}))

    current = {"faithfulness": 0.70}
    failures = _check_regression(current, str(baseline))

    assert len(failures) == 1
    assert "faithfulness" in failures[0]
    assert "0.1000" in failures[0]


def test_check_regression_detects_relevancy_drop(tmp_path: Path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"answer_relevancy": 0.90}))

    current = {"answer_relevancy": 0.80}
    failures = _check_regression(current, str(baseline))

    assert any("answer_relevancy" in f for f in failures)


def test_check_regression_custom_threshold(tmp_path: Path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"faithfulness": 0.80}))

    current = {"faithfulness": 0.77}
    # Default threshold 0.05 → no failure; custom 0.02 → failure
    assert _check_regression(current, str(baseline)) == []
    assert len(_check_regression(current, str(baseline), faithfulness_drop=0.02)) == 1


def test_check_regression_skips_missing_metric_in_baseline(tmp_path: Path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({}))

    failures = _check_regression({"faithfulness": 0.5}, str(baseline))
    assert failures == []


# ---------------------------------------------------------------------------
# run_ragas_eval — import error path (ragas not installed)
# ---------------------------------------------------------------------------


def test_run_ragas_eval_raises_import_error_without_ragas():
    samples = [
        RagasSample(
            question="Q?",
            answer="A.",
            contexts=["Context."],
            ground_truth="G.",
        )
    ]
    with patch.dict(sys.modules, {"ragas": None}):
        with pytest.raises(ImportError, match="pip install ragas"):
            run_ragas_eval(samples)


# ---------------------------------------------------------------------------
# run_ragas_eval — happy path (ragas mocked)
# ---------------------------------------------------------------------------


def test_run_ragas_eval_returns_metric_scores():
    samples = [
        RagasSample(
            question="What is FAISS?",
            answer="FAISS is a library.",
            contexts=["FAISS is a library for similarity search."],
            ground_truth="FAISS is a library.",
        )
    ]

    mock_result = {"faithfulness": 0.9, "answer_relevancy": 0.85}
    mock_metric_faithfulness = MagicMock()
    mock_metric_faithfulness.name = "faithfulness"
    mock_metric_relevancy = MagicMock()
    mock_metric_relevancy.name = "answer_relevancy"

    mock_ragas = MagicMock()
    mock_ragas.EvaluationDataset = MagicMock()
    mock_ragas.SingleTurnSample = MagicMock()
    mock_ragas.evaluate = MagicMock(return_value=mock_result)
    mock_ragas.metrics.Faithfulness = MagicMock(return_value=mock_metric_faithfulness)
    mock_ragas.metrics.AnswerRelevancy = MagicMock(return_value=mock_metric_relevancy)
    mock_ragas.metrics.ContextPrecision = MagicMock()
    mock_ragas.metrics.ContextRecall = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "ragas": mock_ragas,
            "ragas.metrics": mock_ragas.metrics,
        },
    ):
        scores = run_ragas_eval(samples, metrics=("faithfulness", "answer_relevancy"))

    assert scores["faithfulness"] == pytest.approx(0.9)
    assert scores["answer_relevancy"] == pytest.approx(0.85)


def test_run_ragas_eval_empty_metrics_tuple_returns_empty():
    samples = [
        RagasSample(question="Q?", answer="A.", contexts=["C."], ground_truth="G.")
    ]

    mock_ragas = MagicMock()
    mock_ragas.EvaluationDataset = MagicMock()
    mock_ragas.SingleTurnSample = MagicMock()
    mock_ragas.evaluate = MagicMock()
    mock_ragas.metrics = MagicMock()

    with patch.dict(
        sys.modules, {"ragas": mock_ragas, "ragas.metrics": mock_ragas.metrics}
    ):
        scores = run_ragas_eval(samples, metrics=())

    assert scores == {}
    mock_ragas.evaluate.assert_not_called()
