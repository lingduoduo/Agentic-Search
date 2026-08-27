"""Diagnostic benchmarks for the GRPO reward and online-training hot paths.

Optimization in this repository is evidence-driven: a change to a hot path
lands only when a repeatable measurement supports it.  This script produces
that evidence.  It is deliberately *not* a test — it makes no absolute-time
assertion, because a wall-clock threshold on a shared machine fails for reasons
that have nothing to do with the code under measurement.

Run it before and after an optimization with identical flags::

    python -m examples.benchmark_grpo_optimization \\
        --warmup 5 --iterations 25 \\
        --output docs/benchmarks/grpo-optimization-baseline.md \\
        --heading Baseline

Each run appends one Markdown section, so a single file accumulates the
before/after tables for the whole piece of work.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.agents.core.base import AgentLoopOutput
from src.context.search import AgentContext, SearchResult
from src.model.post_training.grpo.algorithms import (
    GRPOAdvantageConfig,
    GRPORolloutSample,
    compute_grpo_outcome_advantage,
    score_prompt_group,
)
from src.model.post_training.log_probs import get_response_log_probs
from src.model.post_training.reward import (
    SearchRewardConfig,
    SearchRewardFunction,
    token_f1_score,
)

# The names a report must contain for a before/after comparison to be
# meaningful. Dropping one silently would turn a missing measurement into an
# apparent improvement.
REQUIRED_CASES = (
    "group_advantages",
    "group_advantages_normalized",
    "group_advantages_tensor",
    "token_f1",
    "reward_components_shaped",
    "reward_components_sparse",
    "reward_batch",
    "reward_token_advantages",
    "score_prompt_group",
    "response_log_probs",
    "training_batch_assembly",
)

# Fixture dimensions. Recorded in every report: a median is meaningless
# without the input size that produced it.
NUM_GROUPS = 16
ROLLOUTS_PER_GROUP = 8
NUM_ROLLOUTS = NUM_GROUPS * ROLLOUTS_PER_GROUP
RESPONSE_LEN = 96
PROMPT_LEN = 32
NUM_ROUNDS = 3
QUERIES_PER_ROUND = 2
DOCS_PER_QUERY = 4
SEED = 20260827


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured case: repeated wall-clock samples plus peak allocation."""

    name: str
    warmup: int
    iterations: int
    samples_ns: tuple[int, ...]
    peak_bytes: int

    @property
    def median_ns(self) -> int:
        return int(statistics.median(self.samples_ns))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "median_ns": self.median_ns,
            "min_ns": min(self.samples_ns),
            "max_ns": max(self.samples_ns),
            "peak_bytes": self.peak_bytes,
        }


def measure_case(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Time *fn* over *iterations* samples after *warmup* unrecorded calls.

    Warmups are excluded from the samples on purpose: the first call pays for
    lazy imports, tokenizer setup and allocator growth, none of which the
    steady-state hot path pays again.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    for _ in range(warmup):
        fn()

    tracemalloc.start()
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        warmup=warmup,
        iterations=iterations,
        samples_ns=tuple(samples),
        peak_bytes=peak_bytes,
    )


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------


def _answer_text() -> str:
    """An answer citing roughly half the retrieved documents."""
    keys = " ".join(
        f"[R{r}Q{q}D{d}]"
        for r in range(1, NUM_ROUNDS + 1)
        for q in range(1, QUERIES_PER_ROUND + 1)
        for d in range(1, DOCS_PER_QUERY // 2 + 1)
    )
    return f"Dense retrieval encodes text into vectors. {keys}"


def _build_context() -> AgentContext:
    ctx = AgentContext()
    for round_idx in range(NUM_ROUNDS):
        queries = [f"query {round_idx}-{q}" for q in range(QUERIES_PER_ROUND)]
        results = [
            [
                SearchResult(
                    contents=f"passage {round_idx}-{q}-{d}",
                    score=1.0 / (d + 1),
                    title=f"Doc {d}",
                    url=f"https://example.test/{round_idx}/{q}/{d}",
                )
                for d in range(DOCS_PER_QUERY)
            ]
            for q in range(QUERIES_PER_ROUND)
        ]
        ctx.add_round(queries, results)
    ctx.record_fetched_pages(
        [SearchResult(contents="fetched", url="https://example.test/0/0/0")]
    )
    return ctx


def _build_outputs(rng: torch.Generator) -> list[AgentLoopOutput]:
    answer = _answer_text()
    ctx = _build_context()
    outputs: list[AgentLoopOutput] = []
    for index in range(NUM_ROLLOUTS):
        outputs.append(
            AgentLoopOutput(
                prompt_ids=list(range(PROMPT_LEN)),
                response_ids=list(range(RESPONSE_LEN)),
                response_mask=[1] * RESPONSE_LEN,
                num_turns=NUM_ROUNDS,
                final_answer=answer,
                context=ctx,
                group_id=f"g{index // ROLLOUTS_PER_GROUP}",
                rollout_index=index % ROLLOUTS_PER_GROUP,
                metrics={
                    "rounds_used": float(NUM_ROUNDS),
                    "search_rounds": float(NUM_ROUNDS),
                    "repeated_search_queries": 1.0,
                    "fetched_pages": 1.0,
                    "unnecessary_fetch_count": 1.0,
                    "web_searches": 1.0,
                    "vdb_searches": 2.0,
                    "rerank_calls": 1.0,
                    "search_quality_score": 0.6,
                    "final_evidence_sufficient": 1.0,
                    "subquestion_coverage_ratio": 0.8,
                    "evidence_gain_total": 0.3,
                    "early_stops": 1.0,
                    "answer_when_evidence_insufficient": 0.0,
                    "forced_final_answer": 0.0,
                    "search_budget_exhausted_without_answer": 0.0,
                    "answer_allowed": 1.0,
                },
            )
        )
    _ = rng
    return outputs


class _PositionLogitModel(torch.nn.Module):
    """Deterministic causal-LM stand-in: logits depend only on position."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        batch, length = input_ids.shape
        base = torch.arange(length, dtype=torch.float32).unsqueeze(-1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(0)
        logits = torch.sin(base + vocab).unsqueeze(0).expand(batch, -1, -1)
        return type("Output", (), {"logits": logits})()


def _build_fixtures() -> dict[str, Any]:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)

    outputs = _build_outputs(rng)
    ground_truths = ["Dense retrieval encodes text into vectors."] * NUM_ROLLOUTS
    group_ids = [output.group_id or "" for output in outputs]
    rewards = [float(torch.rand(1, generator=rng).item()) for _ in range(NUM_ROLLOUTS)]

    vocab_size = 64
    full_ids = torch.randint(
        0, vocab_size, (ROLLOUTS_PER_GROUP, PROMPT_LEN + RESPONSE_LEN), generator=rng
    )
    response_mask = torch.ones(ROLLOUTS_PER_GROUP, RESPONSE_LEN, dtype=torch.long)

    token_rewards = torch.zeros(NUM_ROLLOUTS, RESPONSE_LEN)
    token_rewards[:, -1] = torch.tensor(rewards)
    eos_mask = torch.ones(NUM_ROLLOUTS, RESPONSE_LEN)
    index = torch.arange(NUM_ROLLOUTS) // ROLLOUTS_PER_GROUP

    samples = [
        GRPORolloutSample(
            group_id=f"g{i // ROLLOUTS_PER_GROUP}",
            rollout_index=i % ROLLOUTS_PER_GROUP,
            sampling_params={"temperature": 1.0},
            output=outputs[i],
        )
        for i in range(ROLLOUTS_PER_GROUP)
    ]

    return {
        "outputs": outputs,
        "ground_truths": ground_truths,
        "group_ids": group_ids,
        "rewards": rewards,
        "full_ids": full_ids,
        "response_mask": response_mask,
        "model": _PositionLogitModel(vocab_size),
        "token_rewards": token_rewards,
        "eos_mask": eos_mask,
        "index": index,
        "samples": samples,
        "answer": _answer_text(),
    }


def _fixture_metadata() -> dict[str, int]:
    return {
        "num_groups": NUM_GROUPS,
        "rollouts_per_group": ROLLOUTS_PER_GROUP,
        "num_rollouts": NUM_ROLLOUTS,
        "prompt_len": PROMPT_LEN,
        "response_len": RESPONSE_LEN,
        "retrieved_docs": NUM_ROUNDS * QUERIES_PER_ROUND * DOCS_PER_QUERY,
        "seed": SEED,
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _build_cases(f: dict[str, Any]) -> list[tuple[str, Callable[[], object]]]:
    shaped = SearchRewardFunction()
    sparse = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    judge = token_f1_score
    outputs = f["outputs"]
    truths = f["ground_truths"]

    def _training_batch_assembly() -> object:
        # The prompt-side left-pad in collate_scored_rollouts_for_training,
        # isolated from the generation manager so the measurement is about
        # tensor construction and not about mocking an agent loop.
        rows = [torch.tensor(o.prompt_ids, dtype=torch.long) for o in outputs]
        width = max(row.numel() for row in rows)
        return torch.tensor(
            [[0] * (width - row.numel()) + row.tolist() for row in rows],
            dtype=torch.long,
        )

    return [
        (
            "group_advantages",
            lambda: shaped.compute_grpo_outcome_advantages(
                f["rewards"], f["group_ids"]
            ),
        ),
        (
            "group_advantages_normalized",
            lambda: shaped.compute_batch_advantages(f["rewards"], f["group_ids"]),
        ),
        (
            "group_advantages_tensor",
            lambda: compute_grpo_outcome_advantage(
                f["token_rewards"], f["eos_mask"], f["index"]
            ),
        ),
        ("token_f1", lambda: [token_f1_score(f["answer"], gt) for gt in truths]),
        (
            "reward_components_shaped",
            lambda: [
                shaped.reward_components(o, gt, judge) for o, gt in zip(outputs, truths)
            ],
        ),
        (
            "reward_components_sparse",
            lambda: [
                sparse.reward_components(o, gt, judge) for o, gt in zip(outputs, truths)
            ],
        ),
        (
            "reward_batch",
            lambda: shaped.compute_batch_sparse_token_rewards(outputs, truths, judge),
        ),
        (
            "reward_token_advantages",
            lambda: shaped.assign_grpo_outcome_token_advantages(
                outputs, truths, judge, f["group_ids"]
            ),
        ),
        (
            "score_prompt_group",
            lambda: score_prompt_group(
                f["samples"],
                ground_truth=truths[0],
                judge_fn=judge,
                reward_fn=shaped,
                advantage_config=GRPOAdvantageConfig(),
            ),
        ),
        (
            "response_log_probs",
            lambda: get_response_log_probs(
                f["model"], f["full_ids"], PROMPT_LEN, f["response_mask"]
            ),
        ),
        ("training_batch_assembly", _training_batch_assembly),
    ]


def run_smoke_benchmarks(*, warmup: int = 3, iterations: int = 15) -> dict[str, Any]:
    """Measure every case and return a JSON-serializable report payload."""
    fixtures = _build_fixtures()
    results = [
        measure_case(name, fn, warmup=warmup, iterations=iterations)
        for name, fn in _build_cases(fixtures)
    ]
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "warmup": warmup,
        "iterations": iterations,
        "fixtures": _fixture_metadata(),
        "cases": [result.as_dict() for result in results],
    }


def format_markdown_report(payload: dict[str, Any], *, heading: str) -> str:
    """Render one report section: environment, fixtures, then a case table."""
    fixtures = ", ".join(f"{k}={v}" for k, v in payload["fixtures"].items())
    lines = [
        f"## {heading}",
        "",
        f"- Python {payload['python_version']}, torch {payload['torch_version']}",
        f"- Platform: {payload['platform']} ({payload['device']})",
        f"- Warmup {payload['warmup']}, iterations {payload['iterations']}",
        f"- Fixtures: {fixtures}",
        "",
        "| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| `{case['name']}` "
            f"| {case['median_ns'] / 1000:.1f} "
            f"| {case['min_ns'] / 1000:.1f} "
            f"| {case['max_ns'] / 1000:.1f} "
            f"| {case['peak_bytes'] / 1024:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO optimization benchmarks.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", help="Markdown file to append a report section to.")
    parser.add_argument("--heading", default="Baseline")
    parser.add_argument("--json", help="Optional path for the raw JSON payload.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_smoke_benchmarks(warmup=args.warmup, iterations=args.iterations)
    report = format_markdown_report(payload, heading=args.heading)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = out.read_text() if out.exists() else ""
        separator = "" if not existing or existing.endswith("\n\n") else "\n"
        out.write_text(existing + separator + report)
    if args.json:
        json_out = Path(args.json)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.output:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
