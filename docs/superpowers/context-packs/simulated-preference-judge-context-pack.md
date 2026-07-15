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

…

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

…

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

…

### Task 2: `judge_gold_agreement` validation helper

**Files:**
- Modify: `src/training/judge.py` (append module-level function)
- Modify: `src/training/__init__.py` (add export)
- Test: `tests/unit/test_simulated_judge.py` (append tests)

**Interfaces:**
- Produces: `judge_gold_agreement(pairs: list[tuple[float, bool]]) -> dict[str, float]` returning keys `mean_score_correct`, `mean_score_incorrect`, `gap`, `n_correct`, `n_incorrect`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_simulated_judge.py`:

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_simulated_judge.py -k agreement -v`
Expected: FAIL with `ImportError: cannot import name 'judge_gold_agreement'`

…

### Task 3: GRPO integration test (proves the seam)

This task adds no production code — it proves the judge drives real GRPO advantages through the existing `score_prompt_group` seam, and guards against regressions in that wiring.

**Files:**
- Test: `tests/unit/test_simulated_judge.py` (append)

**Interfaces:**
- Consumes: `GRPORolloutSample` from `src/training/grpo.py`, `AgentLoopOutput` from `src/agents/core/base.py`, `score_prompt_group` from `src/training/grpo.py`.
- `AgentLoopOutput` required constructor args: `prompt_ids: list[int]`, `response_ids: list[int]`, `response_mask: list[int]`, `num_turns: int`; `final_answer` is an optional field.

…

### Final verification

- [ ] Run the whole new test file: `pytest tests/unit/test_simulated_judge.py -v` → all PASS.
- [ ] Run the broader training suite for regressions: `pytest tests/unit -k "grpo or judge or reward" -v` → all PASS.
- [ ] Lint: `ruff check src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py --fix && ruff format src/training/judge.py examples/run_bamboogle_synthetic_grpo.py tests/unit/test_simulated_judge.py`

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
