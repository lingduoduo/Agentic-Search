# Simulated Preference Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, reference-free simulated pointwise judge that stands in for an LLM-as-judge, feeding synthetic AI-feedback scores into GRPO through the existing `batch_judge_fn` seam, demonstrated on the bamboogle seed prompts with a JSONL dump.

**Architecture:** A new `SimulatedPreferenceJudge` scores an answer's quality from text alone and exposes `as_batch_judge_fn()`, matching the `BatchJudgeFn` type GRPO already consumes in `score_prompt_group`/`score_prompt_batch`. No GRPO core changes. A runnable example samples rollouts on bamboogle prompts, scores them with the judge, dumps a synthetic-preference JSONL, and prints a judge-vs-gold agreement report. The real LLM judge is a later drop-in behind the same interface.

**Tech Stack:** Python 3, standard library (`hashlib`), pytest. Reuses `src/training/grpo.py`, `src/training/reward.py`, `src/training/eval/bamboogle.py`, and the server/agent wiring in `examples/run_bamboogle_eval.py`.

## Global Constraints

- **Reference-free judge:** the judge scores from the answer string only and MUST ignore the `ground_truths` argument. Gold answers are used only for the validation report, never as a judge input.
- **Deterministic:** identical answers MUST produce identical scores. Any tie-break jitter derives from `hashlib.sha256`, never the salted built-in `hash()` (which varies per process via `PYTHONHASHSEED`) and never `random`.
- **Scores clamped to `[0, 1]`.**
- **No changes to `src/training/grpo.py` or `src/training/reward.py`** — the `batch_judge_fn` seam already exists.
- **No reward-model training** (flavor B is out of scope).
- Follow the existing `try/except ImportError` export pattern in `src/training/__init__.py`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/training/judge.py` (new) | `SimulatedPreferenceJudge` (the labeler) + `judge_gold_agreement` (validation summary). Light imports only. |
| `examples/run_bamboogle_synthetic_grpo.py` (new) | Orchestration: sample rollouts → score with judge → build records → dump JSONL → print agreement. Contains the pure `build_synthetic_record` helper. |
| `tests/unit/test_simulated_judge.py` (new) | All unit + integration tests for the above. |
| `src/training/__init__.py` (modify) | Export `SimulatedPreferenceJudge` and `judge_gold_agreement`. |

---

## Task 1: `SimulatedPreferenceJudge` core + export

**Files:**
- Create: `src/training/judge.py`
- Modify: `src/training/__init__.py` (add exports inside the existing `try:` block, after the `from .reward import ...` lines)
- Test: `tests/unit/test_simulated_judge.py`

**Interfaces:**
- Consumes: `BatchJudgeFn` from `src/training/reward.py` (type alias `Callable[[list[str], list[str]], list[float]]`).
- Produces:
  - `SimulatedPreferenceJudge` dataclass with `max_words: int = 40`, `jitter_scale: float = 0.05`.
  - `SimulatedPreferenceJudge.score(answer: str) -> float`
  - `SimulatedPreferenceJudge.as_batch_judge_fn() -> BatchJudgeFn`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_simulated_judge.py`:

```python
from src.training.judge import SimulatedPreferenceJudge


def test_empty_answer_scores_zero():
    judge = SimulatedPreferenceJudge()
    assert judge.score("") == 0.0
    assert judge.score("   ") == 0.0


def test_score_is_deterministic():
    judge = SimulatedPreferenceJudge()
    a = "James Madison was president when Citibank was founded"
    assert judge.score(a) == judge.score(a)


def test_scores_stay_in_unit_interval():
    judge = SimulatedPreferenceJudge()
    for answer in ["", "x", "paris " * 200, "James Madison was president"]:
        s = judge.score(answer)
        assert 0.0 <= s <= 1.0


def test_varied_answer_beats_degenerate_repetition():
    judge = SimulatedPreferenceJudge()
    varied = "James Madison was the president at that time"
    degenerate = "paris paris paris paris paris paris paris"
    assert judge.score(varied) > judge.score(degenerate)


def test_hedging_answer_is_penalized():
    judge = SimulatedPreferenceJudge()
    concrete = "The answer is James Madison the fourth president"
    hedge = "I don't know the answer to this question really"
    assert judge.score(concrete) > judge.score(hedge)


def test_as_batch_judge_fn_length_and_ignores_ground_truth():
    judge = SimulatedPreferenceJudge()
    fn = judge.as_batch_judge_fn()
    answers = ["James Madison was president", ""]
    scores_no_gt = fn(answers, [])          # ground_truths ignored
    scores_with_gt = fn(answers, ["madison", "madison"])
    assert len(scores_no_gt) == 2
    assert scores_no_gt == scores_with_gt   # ground truth has no effect
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_simulated_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.judge'`

- [ ] **Step 3: Write the implementation**

Create `src/training/judge.py`:

```python
"""Simulated, deterministic pointwise judge standing in for an LLM-as-judge.

The judge scores an answer's quality/form from the answer text alone
(reference-free) and returns a scalar in ``[0, 1]``.  It is a drop-in for the
``BatchJudgeFn`` seam GRPO already consumes (``score_prompt_group`` /
``score_prompt_batch`` in :mod:`src.training.grpo`); a real LLM judge can
replace it behind the same :meth:`SimulatedPreferenceJudge.as_batch_judge_fn`
interface.

Scores are deterministic: identical answers always produce identical scores.
Tie-break jitter is derived from a SHA-256 digest, never the salted built-in
``hash`` (which varies per process) and never ``random`` — so tests and cached
runs are reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.training.reward import BatchJudgeFn

_HEDGES = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "not sure",
    "unknown",
)


@dataclass(frozen=True)
class SimulatedPreferenceJudge:
    """Reference-free, deterministic pointwise answer-quality judge.

    ``max_words``: answers up to this many words get full length credit;
    longer answers are penalized.  ``jitter_scale``: magnitude of the
    deterministic tie-break term added to the base score.
    """

    max_words: int = 40
    jitter_scale: float = 0.05

    def score(self, answer: str) -> float:
        """Return a quality score in ``[0, 1]`` from the answer text alone."""
        text = answer.strip()
        if not text:
            return 0.0
        words = text.split()
        n = len(words)
        if n < 2:
            length_score = 0.3
        elif n <= self.max_words:
            length_score = 1.0
        else:
            length_score = max(0.2, 1.0 - (n - self.max_words) / 100.0)
        unique_ratio = len({w.lower() for w in words}) / n
        lowered = text.lower()
        hedge_penalty = 0.5 if any(h in lowered for h in _HEDGES) else 0.0
        base = 0.5 * length_score + 0.5 * unique_ratio - hedge_penalty
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        jitter = (digest[0] / 255.0) * self.jitter_scale
        return max(0.0, min(1.0, base + jitter))

    def as_batch_judge_fn(self) -> BatchJudgeFn:
        """Adapt to the ``BatchJudgeFn`` GRPO expects (ground truth ignored)."""

        def _judge(answers: list[str], ground_truths: list[str]) -> list[float]:
            return [self.score(a) for a in answers]

        return _judge
```

- [ ] **Step 4: Add the export**

In `src/training/__init__.py`, inside the existing `try:` block, immediately after the line `from .sft import build_search_sft_example as build_search_sft_example`, add:

```python
    from .judge import SimulatedPreferenceJudge as SimulatedPreferenceJudge
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_simulated_judge.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/training/judge.py src/training/__init__.py tests/unit/test_simulated_judge.py
git commit -m "feat(judge): deterministic reference-free SimulatedPreferenceJudge"
```

---

## Task 2: `judge_gold_agreement` validation helper

**Files:**
- Modify: `src/training/judge.py` (append module-level function)
- Modify: `src/training/__init__.py` (add export)
- Test: `tests/unit/test_simulated_judge.py` (append tests)

**Interfaces:**
- Produces: `judge_gold_agreement(pairs: list[tuple[float, bool]]) -> dict[str, float]` returning keys `mean_score_correct`, `mean_score_incorrect`, `gap`, `n_correct`, `n_incorrect`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_simulated_judge.py`:

```python
from src.training.judge import judge_gold_agreement


def test_agreement_gap_positive_when_correct_scores_higher():
    pairs = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    report = judge_gold_agreement(pairs)
    assert report["mean_score_correct"] == 0.85
    assert report["mean_score_incorrect"] == 0.15
    assert report["gap"] > 0
    assert report["n_correct"] == 2.0
    assert report["n_incorrect"] == 2.0


def test_agreement_handles_all_correct():
    pairs = [(0.7, True), (0.9, True)]
    report = judge_gold_agreement(pairs)
    assert report["mean_score_correct"] == 0.8
    assert report["mean_score_incorrect"] == 0.0
    assert report["gap"] == 0.8


def test_agreement_handles_empty_input():
    report = judge_gold_agreement([])
    assert report["gap"] == 0.0
    assert report["n_correct"] == 0.0
    assert report["n_incorrect"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_simulated_judge.py -k agreement -v`
Expected: FAIL with `ImportError: cannot import name 'judge_gold_agreement'`

- [ ] **Step 3: Write the implementation**

Append to `src/training/judge.py`:

```python
def judge_gold_agreement(pairs: list[tuple[float, bool]]) -> dict[str, float]:
    """Summarise how well judge scores separate correct from incorrect answers.

    ``pairs`` is a list of ``(judge_score, is_correct)``.  A positive ``gap``
    means the judge scores correct answers higher on average — evidence the
    (simulated) judge tracks correctness rather than being nonsense.  On hard
    factual questions a reference-free judge may show a small or zero gap; that
    is an informative result, not a failure.
    """
    correct = [s for s, ok in pairs if ok]
    incorrect = [s for s, ok in pairs if not ok]
    mean_correct = sum(correct) / len(correct) if correct else 0.0
    mean_incorrect = sum(incorrect) / len(incorrect) if incorrect else 0.0
    return {
        "mean_score_correct": mean_correct,
        "mean_score_incorrect": mean_incorrect,
        "gap": mean_correct - mean_incorrect,
        "n_correct": float(len(correct)),
        "n_incorrect": float(len(incorrect)),
    }
```

- [ ] **Step 4: Add the export**

In `src/training/__init__.py`, immediately after the `from .judge import SimulatedPreferenceJudge ...` line added in Task 1, add:

```python
    from .judge import judge_gold_agreement as judge_gold_agreement
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_simulated_judge.py -k agreement -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/training/judge.py src/training/__init__.py tests/unit/test_simulated_judge.py
git commit -m "feat(judge): judge_gold_agreement validation summary"
```

---

## Task 3: GRPO integration test (proves the seam)

This task adds no production code — it proves the judge drives real GRPO advantages through the existing `score_prompt_group` seam, and guards against regressions in that wiring.

**Files:**
- Test: `tests/unit/test_simulated_judge.py` (append)

**Interfaces:**
- Consumes: `GRPORolloutSample` from `src/training/grpo.py`, `AgentLoopOutput` from `src/agents/core/base.py`, `score_prompt_group` from `src/training/grpo.py`.
- `AgentLoopOutput` required constructor args: `prompt_ids: list[int]`, `response_ids: list[int]`, `response_mask: list[int]`, `num_turns: int`; `final_answer` is an optional field.
- `score_prompt_group(samples, *, ground_truth, judge_fn, reward_fn=None, advantage_config=None, batch_judge_fn=None, metadata=None)` — `judge_fn` is required positional-by-keyword even when `batch_judge_fn` wins, so pass a trivial stub.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_simulated_judge.py`:

```python
from src.agents.core.base import AgentLoopOutput
from src.training.grpo import GRPORolloutSample, score_prompt_group


def _fake_sample(group_id: str, idx: int, answer: str) -> GRPORolloutSample:
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=answer,
    )
    return GRPORolloutSample(
        group_id=group_id,
        rollout_index=idx,
        sampling_params={},
        output=output,
    )


def test_sim_judge_drives_nondegenerate_grpo_advantages():
    judge = SimulatedPreferenceJudge()
    samples = [
        _fake_sample("g", 0, "James Madison was the president at that time"),
        _fake_sample("g", 1, ""),
        _fake_sample("g", 2, "paris paris paris paris paris paris"),
    ]
    scored = score_prompt_group(
        samples,
        ground_truth="james madison",
        judge_fn=lambda pred, gold: 0.0,
        batch_judge_fn=judge.as_batch_judge_fn(),
    )
    advantages = [s.advantage for s in scored]
    assert len(advantages) == 3
    # Not all advantages collapse to zero — the judge produced a real spread.
    assert any(abs(a) > 1e-6 for a in advantages)
    # The empty answer must not be the best-advantaged rollout.
    assert scored[1].advantage < max(advantages)
```

- [ ] **Step 2: Run the test to verify it passes immediately**

Run: `pytest tests/unit/test_simulated_judge.py::test_sim_judge_drives_nondegenerate_grpo_advantages -v`
Expected: PASS (the seam already exists; this test confirms it). If it FAILS, do not modify `grpo.py` — investigate whether `batch_judge_fn` is being threaded and stop to report.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_simulated_judge.py
git commit -m "test(judge): sim judge drives non-degenerate GRPO advantages via existing seam"
```

---

## Task 4: `run_bamboogle_synthetic_grpo.py` demo + JSONL dump

**Files:**
- Create: `examples/run_bamboogle_synthetic_grpo.py`
- Test: `tests/unit/test_simulated_judge.py` (append test for the pure record builder)

**Interfaces:**
- Consumes: `SimulatedPreferenceJudge` (Task 1), `ScoredGRPORollout` from `src/training/grpo.py` (fields: `rollout_index`, `output.final_answer`, `reward`, `advantage`), `contains_match` from `src/training/eval/bamboogle.py`, and the server/agent wiring from `examples/run_bamboogle_eval.py` (`_build_server_manager`, `SearchAgentLoop`, `SearchAgentLoopConfig`, `SearchEvaluationConfig`).
- Produces: `build_synthetic_record(prompt: str, gold: list[str], judge: SimulatedPreferenceJudge, scored: list[ScoredGRPORollout]) -> dict` — one synthetic-preference record.

- [ ] **Step 1: Write the failing test for the record builder**

Append to `tests/unit/test_simulated_judge.py`:

```python
from src.training.grpo import ScoredGRPORollout


def _scored(idx: int, answer: str, reward: float, advantage: float) -> ScoredGRPORollout:
    output = AgentLoopOutput(
        prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
        final_answer=answer,
    )
    return ScoredGRPORollout(
        group_id="g",
        rollout_index=idx,
        sampling_params={},
        output=output,
        reward=reward,
        reward_component="total",
        reward_components={"correctness": reward},
        advantage=advantage,
    )


def test_build_synthetic_record_schema():
    from examples.run_bamboogle_synthetic_grpo import build_synthetic_record

    judge = SimulatedPreferenceJudge()
    scored = [
        _scored(0, "James Madison", 0.9, 0.4),
        _scored(1, "", 0.0, -0.4),
    ]
    record = build_synthetic_record(
        prompt="Who was president when Citibank was founded?",
        gold=["james madison"],
        judge=judge,
        scored=scored,
    )
    assert record["prompt"] == "Who was president when Citibank was founded?"
    assert record["gold"] == ["james madison"]
    assert len(record["rollouts"]) == 2
    first = record["rollouts"][0]
    assert set(first) == {
        "answer", "judge_score", "reward", "advantage",
        "exact_match", "contains_match",
    }
    assert first["contains_match"] == 1.0        # "James Madison" contains gold
    assert first["judge_score"] == judge.score("James Madison")
    assert record["rollouts"][1]["contains_match"] == 0.0  # empty answer
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_simulated_judge.py::test_build_synthetic_record_schema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.run_bamboogle_synthetic_grpo'`

- [ ] **Step 3: Write the example script**

Create `examples/run_bamboogle_synthetic_grpo.py`:

```python
"""Generate synthetic AI-feedback on Bamboogle prompts and feed it to GRPO.

For each Bamboogle prompt this samples ``--num_rollouts`` agent answers, scores
them with the reference-free :class:`SimulatedPreferenceJudge` (a stand-in for
an LLM-as-judge), computes GRPO group-relative advantages via the existing
``score_prompt_group`` seam, dumps a synthetic-preference dataset to JSONL, and
prints a judge-vs-gold agreement report.

The judge is the only simulated piece: prompts are real Bamboogle questions and
answers are real model rollouts.  Swap ``SimulatedPreferenceJudge`` for a real
LLM judge behind the same ``as_batch_judge_fn()`` interface to go live.

Quick start (local CPU, self-contained, slow):
    python3 -m examples.run_bamboogle_synthetic_grpo \\
        --model Qwen/Qwen2.5-1.5B-Instruct --local \\
        --search_url http://localhost:8000/retrieve --num_rollouts 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.training.eval.bamboogle import contains_match, exact_match, load_bamboogle
from src.training.grpo import (
    ScoredGRPORollout,
    sample_prompt_group,
    score_prompt_group,
)
from src.training.judge import SimulatedPreferenceJudge, judge_gold_agreement


def build_synthetic_record(
    prompt: str,
    gold: list[str],
    judge: SimulatedPreferenceJudge,
    scored: list[ScoredGRPORollout],
) -> dict[str, Any]:
    """Build one synthetic-preference JSONL record for a prompt group."""
    rollouts = []
    for s in scored:
        answer = s.output.final_answer or ""
        rollouts.append(
            {
                "answer": answer,
                "judge_score": judge.score(answer),
                "reward": s.reward,
                "advantage": s.advantage,
                "exact_match": exact_match(answer, gold),
                "contains_match": contains_match(answer, gold),
            }
        )
    return {"prompt": prompt, "gold": gold, "rollouts": rollouts}


def _build_loop_factory(args: argparse.Namespace, tokenizer: Any):
    from examples.run_bamboogle_eval import _build_server_manager
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    from src.training.evaluation import SearchEvaluationConfig

    server_manager = _build_server_manager(args, tokenizer)

    def factory() -> SearchAgentLoop:
        return SearchAgentLoop(
            tokenizer=tokenizer,
            server_manager=server_manager,
            search_config=SearchAgentLoopConfig(
                search_url=args.search_url,
                topk=args.topk,
                max_turns=args.max_turns,
                evaluation_config=SearchEvaluationConfig(
                    min_results_per_query=1,
                    min_total_results=2,
                    min_content_length=10,
                ),
            ),
        )

    return factory, server_manager


async def _run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not args.allow_remote_model_downloads,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    judge = SimulatedPreferenceJudge()
    loop_factory, server_manager = _build_loop_factory(args, tokenizer)
    examples = load_bamboogle(limit=args.limit)

    pairs: list[tuple[float, bool]] = []
    out_path = Path(args.output)
    try:
        with out_path.open("w") as fh:
            for ex in examples:
                question = ex["question"]
                gold = ex.get("golden_answers") or ex.get("answers") or []
                samples = await sample_prompt_group(
                    loop_factory,
                    messages=[{"role": "user", "content": question}],
                    sampling_params={
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                    },
                    num_rollouts=args.num_rollouts,
                )
                scored = score_prompt_group(
                    samples,
                    ground_truth=gold[0] if gold else "",
                    judge_fn=lambda pred, g: 0.0,
                    batch_judge_fn=judge.as_batch_judge_fn(),
                )
                record = build_synthetic_record(question, gold, judge, scored)
                fh.write(json.dumps(record) + "\n")
                for r in record["rollouts"]:
                    pairs.append((r["judge_score"], r["contains_match"] > 0))
                print(f"[{question[:60]}...] {len(scored)} rollouts scored")
    finally:
        await server_manager.aclose()

    report = judge_gold_agreement(pairs)
    print(f"\nSynthetic dataset written to {out_path}")
    print("Judge-vs-gold agreement:")
    for k, v in report.items():
        print(f"  {k:20s}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic AI-feedback on Bamboogle for GRPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--local", action="store_true", help="Run model in-process")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow_unsafe_mps", action="store_true")
    parser.add_argument("--allow_remote_model_downloads", action="store_true")
    parser.add_argument("--server_url", default="http://localhost:8080")
    parser.add_argument("--search_url", default="http://localhost:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts")
    parser.add_argument("--output", default="bamboogle_synthetic.jsonl")
    parser.add_argument("--generation_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--generation_heartbeat_seconds", type=float, default=10.0)
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
```

> Note: `_build_server_manager` in `examples/run_bamboogle_eval.py` reads
> `args.generation_timeout_seconds` and `args.generation_heartbeat_seconds` in
> the `--local` branch, which is why those two args are declared here.

- [ ] **Step 4: Run the record-builder test to verify it passes**

Run: `pytest tests/unit/test_simulated_judge.py::test_build_synthetic_record_schema -v`
Expected: PASS

- [ ] **Step 5: Run the full test file**

Run: `pytest tests/unit/test_simulated_judge.py -v`
Expected: PASS (all tests from Tasks 1-4)

- [ ] **Step 6: Verify the example imports and its CLI is wired**

Run: `python3 -m examples.run_bamboogle_synthetic_grpo --help`
Expected: argparse help text prints with `--model`, `--num_rollouts`, `--output`, and exit code 0.

- [ ] **Step 7: Commit**

```bash
git add examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py
git commit -m "feat(judge): bamboogle synthetic AI-feedback demo with JSONL dump + agreement"
```

---

## Final verification

- [ ] Run the whole new test file: `pytest tests/unit/test_simulated_judge.py -v` → all PASS.
- [ ] Run the broader training suite for regressions: `pytest tests/unit -k "grpo or judge or reward" -v` → all PASS.
- [ ] Lint: `ruff check src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py --fix && ruff format src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py`

---

## Self-Review

**Spec coverage:**
- Pointwise reference-free judge → Task 1 (`score`, `as_batch_judge_fn`, ignores ground truth). ✓
- Deterministic (SHA-256, no `hash`/`random`) → Task 1 impl + `test_score_is_deterministic`. ✓
- Wired into GRPO via existing `batch_judge_fn` seam, no core changes → Task 3 proves it. ✓
- Runnable bamboogle demo → Task 4. ✓
- JSONL synthetic-dataset dump for tracking → Task 4 `build_synthetic_record` + writer, schema test. ✓
- Validation report vs gold with honest caveat → Task 2 `judge_gold_agreement` (+ docstring caveat), used in Task 4. ✓
- Deterministic offline tests → all tests are offline; the only online piece (rollout sampling) lives in the example and is not unit-tested. ✓
- Export follows existing pattern → Tasks 1-2 add exports inside the `try/except ImportError` block. ✓
- Out-of-scope items (real LLM adapter, reward model) → not present. ✓

**Deviation from spec (simplification, flagged):** the spec's optional "citation/evidence support when `loop_output` metadata is present" bullet is dropped. The GRPO `batch_judge_fn` seam only carries answer text + ground truth, so a metadata-aware judge cannot be fed through it; the judge is text-only. This is a simplification, not added scope.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows complete code. ✓

**Type consistency:** `SimulatedPreferenceJudge`, `score`, `as_batch_judge_fn`, `judge_gold_agreement`, `build_synthetic_record` names and signatures are identical across tasks and tests. `AgentLoopOutput`/`GRPORolloutSample`/`ScoredGRPORollout` constructor fields match the real definitions in `src/agents/core/base.py` and `src/training/grpo.py`. ✓
