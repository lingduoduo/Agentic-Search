"""RAGAS evaluation harness for the RAG pipeline.

Evaluates retrieval + generation quality using RAGAS metrics:
  - faithfulness:      answer claims are grounded in retrieved context
  - answer_relevancy:  answer is on-topic for the question
  - context_precision: relevant docs ranked before irrelevant ones
  - context_recall:    retrieved context covers the ground-truth answer

Requires: pip install ragas
OpenAI API key required for LLM-backed metrics (faithfulness, answer_relevancy,
context_recall). Set OPENAI_API_KEY before running.

QA pairs file (one JSON object per line):
    {"question": "What is FAISS?", "ground_truth": "FAISS is a library ..."}

Usage:
    python -m src.internal.retrieval.ragas_eval \\
        --dataset data/eval/ragas_qa.jsonl \\
        [--search_url http://localhost:8001/search] \\
        [--baseline data/eval/ragas_baseline.json] \\
        [--output data/eval/ragas_results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .service import RetrievalService

logger = logging.getLogger(__name__)

_DEFAULT_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_FAITHFULNESS_DROP_THRESHOLD = 0.05
_RELEVANCY_DROP_THRESHOLD = 0.05


@dataclass
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def build_ragas_dataset(
    qa_pairs: list[dict],
    *,
    service: RetrievalService,
    top_k: int = 5,
) -> list[RagasSample]:
    """Retrieve contexts for each QA pair and return RagasSample list.

    Answers are synthesized extractively (first sentence of top context)
    for retrieval-only evaluation. For full answer-quality evaluation, replace
    the answer field with LLM-generated text from the answer_with_retrieval pipeline.
    """
    samples: list[RagasSample] = []
    for item in qa_pairs:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        try:
            results, _ = service.search(question, top_k=top_k)
        except Exception as exc:
            logger.warning("Retrieval failed for %r: %s", question, exc)
            continue
        contexts = [r.text for r in results if r.text]
        answer = contexts[0].split(".")[0].strip() + "." if contexts else ""
        samples.append(
            RagasSample(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        )
    return samples


def run_ragas_eval(
    samples: list[RagasSample],
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> dict[str, float]:
    """Evaluate samples with RAGAS and return per-metric averages.

    Raises ImportError when the `ragas` package is not installed.
    Requires OPENAI_API_KEY for LLM-backed metrics.
    """
    try:
        from ragas import EvaluationDataset  # type: ignore[import]
        from ragas import SingleTurnSample  # type: ignore[import]
        from ragas import evaluate as ragas_evaluate  # type: ignore[import]
        from ragas.metrics import AnswerRelevancy  # type: ignore[import]
        from ragas.metrics import ContextPrecision  # type: ignore[import]
        from ragas.metrics import ContextRecall  # type: ignore[import]
        from ragas.metrics import Faithfulness  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "RAGAS eval requires the 'ragas' package. Install with: pip install ragas"
        ) from exc

    _metric_cls: dict[str, object] = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
    }
    active = [_metric_cls[m] for m in metrics if m in _metric_cls]
    if not active:
        return {}

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.ground_truth,
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)
    result = ragas_evaluate(dataset=dataset, metrics=active)

    scores: dict[str, float] = {}
    for m in active:
        name = m.name
        try:
            scores[name] = float(result[name])
        except (KeyError, TypeError):
            logger.warning("Metric %r not in RAGAS result", name)
    return scores


def _check_regression(
    current: dict[str, float],
    baseline_path: str,
    *,
    faithfulness_drop: float = _FAITHFULNESS_DROP_THRESHOLD,
    relevancy_drop: float = _RELEVANCY_DROP_THRESHOLD,
) -> list[str]:
    """Return regression messages (empty list = no regressions)."""
    path = Path(baseline_path)
    if not path.exists():
        logger.info("No baseline at %s — skipping regression check", baseline_path)
        return []

    baseline: dict = json.loads(path.read_text())
    thresholds = {
        "faithfulness": faithfulness_drop,
        "answer_relevancy": relevancy_drop,
    }
    failures: list[str] = []
    for metric, threshold in thresholds.items():
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            continue
        drop = base_val - curr_val
        if drop > threshold:
            failures.append(
                f"{metric}: dropped {drop:.4f} "
                f"(baseline={base_val:.4f}, current={curr_val:.4f}, threshold={threshold:.4f})"
            )
    return failures


def _load_qa_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for the RAG pipeline"
    )
    parser.add_argument("--dataset", required=True, help="Path to ragas_qa.jsonl")
    parser.add_argument("--search_url", default="http://localhost:8001/search")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(_DEFAULT_METRICS),
        choices=list(_DEFAULT_METRICS),
    )
    parser.add_argument(
        "--output", default=None, help="Write results JSON to this path"
    )
    parser.add_argument(
        "--baseline", default=None, help="Baseline JSON for regression check"
    )
    args = parser.parse_args()

    service = RetrievalService.from_env()
    qa_pairs = _load_qa_pairs(args.dataset)
    logger.info("Building RAGAS dataset from %d QA pairs...", len(qa_pairs))
    samples = build_ragas_dataset(qa_pairs, service=service, top_k=args.top_k)
    logger.info("Running RAGAS eval on %d samples...", len(samples))
    scores = run_ragas_eval(samples, metrics=tuple(args.metrics))

    output_str = json.dumps(scores, indent=2)
    if args.output:
        Path(args.output).write_text(output_str)
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)

    if args.baseline:
        failures = _check_regression(scores, args.baseline)
        if failures:
            print("\nREGRESSION DETECTED:")
            for f in failures:
                print(f"  {f}")
            raise SystemExit(1)
        else:
            print("\nNo regressions detected.")


if __name__ == "__main__":
    main()
