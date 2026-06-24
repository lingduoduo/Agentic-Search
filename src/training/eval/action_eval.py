"""Aggregate action-policy eval metrics and compare baseline vs trained.

The agent loop already emits the action-mix metrics on ``AgentLoopOutput.metrics``
(``web_searches``, ``vdb_searches``, ``search_rounds``, ``rerank_calls``,
``evidence_score_final``). This module turns a set of eval rollouts into an
aggregate and checks the spec's headline success criterion:

    trained mean search_rounds <= baseline  AND  trained mean correctness >= baseline

so a retriever-aware policy is only credited when it spends fewer search rounds
without losing answer quality.

A "sample" is a dict ``{"correctness": float, "metrics": {...}}`` — typically
built per eval question from the judge score and the rollout's ``output.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionEvalAggregate:
    n: int
    mean_correctness: float
    mean_search_rounds: float
    mean_web_searches: float
    mean_vdb_searches: float
    web_fraction: float  # web / (web + vdb) searches; 0 when no searches
    rerank_rate: float  # fraction of samples that invoked >=1 rerank
    mean_evidence_score_final: float


@dataclass(frozen=True)
class EvalComparison:
    baseline: ActionEvalAggregate
    trained: ActionEvalAggregate
    fewer_rounds: bool
    correctness_preserved: bool
    meets_success_criterion: bool


def aggregate_action_metrics(samples: list[dict]) -> ActionEvalAggregate:
    n = len(samples)
    if n == 0:
        return ActionEvalAggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _mean(key: str) -> float:
        return sum(float(s["metrics"].get(key, 0.0)) for s in samples) / n

    mean_web = _mean("web_searches")
    mean_vdb = _mean("vdb_searches")
    total_searches = (mean_web + mean_vdb) * n
    web_total = mean_web * n
    rerank_samples = sum(
        1 for s in samples if float(s["metrics"].get("rerank_calls", 0.0)) > 0.0
    )
    return ActionEvalAggregate(
        n=n,
        mean_correctness=sum(float(s["correctness"]) for s in samples) / n,
        mean_search_rounds=_mean("search_rounds"),
        mean_web_searches=mean_web,
        mean_vdb_searches=mean_vdb,
        web_fraction=(web_total / total_searches) if total_searches else 0.0,
        rerank_rate=rerank_samples / n,
        mean_evidence_score_final=_mean("evidence_score_final"),
    )


def compare_action_evals(
    baseline: ActionEvalAggregate, trained: ActionEvalAggregate
) -> EvalComparison:
    fewer_rounds = trained.mean_search_rounds <= baseline.mean_search_rounds
    correctness_preserved = trained.mean_correctness >= baseline.mean_correctness
    return EvalComparison(
        baseline=baseline,
        trained=trained,
        fewer_rounds=fewer_rounds,
        correctness_preserved=correctness_preserved,
        meets_success_criterion=fewer_rounds and correctness_preserved,
    )


def format_comparison_table(comparison: EvalComparison) -> str:
    b, t = comparison.baseline, comparison.trained
    rows = [
        ("metric", "baseline", "trained"),
        ("n", str(b.n), str(t.n)),
        ("correctness", f"{b.mean_correctness:.3f}", f"{t.mean_correctness:.3f}"),
        ("search_rounds", f"{b.mean_search_rounds:.3f}", f"{t.mean_search_rounds:.3f}"),
        ("web_searches", f"{b.mean_web_searches:.3f}", f"{t.mean_web_searches:.3f}"),
        ("vdb_searches", f"{b.mean_vdb_searches:.3f}", f"{t.mean_vdb_searches:.3f}"),
        ("web_fraction", f"{b.web_fraction:.3f}", f"{t.web_fraction:.3f}"),
        ("rerank_rate", f"{b.rerank_rate:.3f}", f"{t.rerank_rate:.3f}"),
        (
            "evidence_final",
            f"{b.mean_evidence_score_final:.3f}",
            f"{t.mean_evidence_score_final:.3f}",
        ),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [f"{name:<{width}}  {base:>10}  {trn:>10}" for name, base, trn in rows]
    verdict = "PASS" if comparison.meets_success_criterion else "FAIL"
    lines.append(
        f"\nSuccess criterion (fewer rounds AND correctness preserved): {verdict}"
    )
    return "\n".join(lines)
