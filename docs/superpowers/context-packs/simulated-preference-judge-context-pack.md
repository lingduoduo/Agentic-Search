# Generated Context Pack

# Simulated Preference Judge

## Sources

- [Specification: 2026-07-09-simulated-preference-judge-design.md](../specs/2026-07-09-simulated-preference-judge-design.md)
- [Plan: 2026-07-09-simulated-preference-judge.md](../plans/2026-07-09-simulated-preference-judge.md)

## Specification Context

### Non-Goals (YAGNI)

- **No real LLM judge adapter** in this change. The `BatchJudgeFn` interface is
  already threaded through GRPO; a real judge is a later one-liner behind it.
- **No reward-model training** (no Bradley-Terry / value head). That is the
  separate "flavor B" and is explicitly out of scope.
- **No new dataset.** We reuse the existing bamboogle seed prompts.
- **No changes to the GRPO core algorithm.** The judge plugs into the existing
  `batch_judge_fn` parameter of `score_prompt_group` / `score_prompt_batch`.

### Key Decisions

1. **Judge mode: pointwise score.** The judge returns a scalar in `[0, 1]` per
   answer. GRPO performs the group-relative comparison itself
   (`compute_grpo_outcome_advantages`, `src/training/reward.py`). This fits the
   existing `BatchJudgeFn` seam exactly; no ranking schema or chosen/rejected
   pairs are introduced.

2. **Online, function GRPO calls.** The judge is invoked inside the GRPO loop
   (one batched call per prompt group), not an offline cached dataset. It scores
   the *current* policy's fresh rollouts.

3. **Simulated, not real LLM.** A deterministic reference implementation stands
   in for the LLM. Real LLM = drop-in swap behind the same interface.

4. **Reference-free.** The judge scores answer *quality/form* from the answer's
   own features and ignores the `ground_truths` argument. Gold answers are used
   **only for validation** (measuring judge↔correctness agreement), never as a
   judge input.

5. **Seed corpus: bamboogle.** The 5 `data/bamboogle_train/*.parquet` prompts
   are already in the pipeline's expected schema (`data_source='bamboogle'`,
   `reward_model.ground_truth.target`) and small enough that the whole loop runs
   on a laptop (~5 judge calls/epoch).

### Testing (deterministic, offline)

- **Judge stability:** identical inputs → identical scores; empty answer →
  low score; well-formed cited answer → higher score.
- **Batch adapter:** `as_batch_judge_fn()` returns correct length; ignores
  `ground_truths`.
- **GRPO integration:** `score_prompt_group` with the sim judge yields
  non-degenerate (non-all-zero) advantages across a varied group.
- **Validation metric:** agreement computation runs and returns a number on a
  small synthetic set.

## Implementation Plan Context

### Global Constraints

- **Reference-free judge:** the judge scores from the answer string only and MUST ignore the `ground_truths` argument. Gold answers are used only for the validation report, never as a judge input.
- **Deterministic:** identical answers MUST produce identical scores. Any tie-break jitter derives from `hashlib.sha256`, never the salted built-in `hash()` (which varies per process via `PYTHONHASHSEED`) and never `random`.
- **Scores clamped to `[0, 1]`.**
- **No changes to `src/training/grpo.py` or `src/training/reward.py`** — the `batch_judge_fn` seam already exists.
- **No reward-model training** (flavor B is out of scope).
- Follow the existing `try/except ImportError` export pattern in `src/training/__init__.py`.

---

### Task 1: `SimulatedPreferenceJudge` core + export

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

_[Section compacted.]_

### Task 2: `judge_gold_agreement` validation helper

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

_[Section compacted.]_

### Task 3: GRPO integration test (proves the seam)

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

_[Section compacted.]_

### Task 4: `run_bamboogle_synthetic_grpo.py` demo + JSONL dump

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

_[Section compacted.]_

### Final verification

- [ ] Run the whole new test file: `pytest tests/unit/test_simulated_judge.py -v` → all PASS.
- [ ] Run the broader training suite for regressions: `pytest tests/unit -k "grpo or judge or reward" -v` → all PASS.
- [ ] Lint: `ruff check src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py --fix && ruff format src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py`

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
