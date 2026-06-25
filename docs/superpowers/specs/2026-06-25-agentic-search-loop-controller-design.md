# Agentic Search — LoopController (control-flow behavior) design

**Date:** 2026-06-25
**Status:** Approved (design); implementation plan pending
**Scope:** Phase 1 of a two-phase control-flow improvement for `SearchAgentLoop`.

## Goal

Improve the control flow of agentic search by making the loop's two implicit
decisions explicit, and shipping four behavior improvements on top of them:

1. **Plateau early-stop** — stop searching when marginal evidence gain stalls.
2. **Graceful dead-end answer** — emit a best-effort answer at a dead-end instead
   of returning nothing.
3. **Smarter answer-gating** — targeted per-subquestion feedback, then a clean
   forced-answer path.
4. **Adaptive search budget** — scale the search-round budget by subquestion count.

## Sequencing (agreed)

- **This phase (behavior):** the four behaviors above, default-on.
- **Later phase (structure):** the full state-machine refactor + decomposition of
  the ~400-line `run()` and unifying the loop onto the modular components
  (`Planner`/`SearchTool`/...). Out of scope here; the `LoopController` seam is
  designed to feed directly into it.

Rationale for behavior-first inside the un-refactored loop: each behavior change
is kept small and test-pinned; `LoopController` isolates the new policy so the
monolith grows minimally.

## Activation (agreed)

Default-on; test and reward baselines updated in this work. New config knobs ship
with defaults that change behavior, but with low blast radius (see Defaults).

## Key insight

The four behaviors collapse into **two decisions** the loop currently makes in
scattered, implicit ways:

- **Decision A — "should I keep searching?"** = plateau early-stop + adaptive
  budget. Today split across the `rounds_used >= max_search_limit` check and the
  observability-only plateau check at `search.py:873`.
- **Decision B — "how do I produce the final answer?"** = graceful dead-end +
  smarter gating. Today split across the answer-gate (`search.py:1052`), the
  reject-3×-then-force path (`search.py:1068`), and several `break`-with-
  `final_answer=None` exits. Graceful dead-end and forced-after-rejection are the
  **same mechanism**: a single forced-final-answer turn.

## Architecture (Approach 2: extract a `LoopController`)

**New file:** `src/agents/components/loop_controller.py`. It owns the *policy* for
the two decisions; `run()` remains the *state owner* and consults it. The
controller is pure-over-snapshot (no I/O, no model calls), holding only `cfg`.

```python
@dataclass(frozen=True)
class LoopSnapshot:
    rounds_used: int
    num_subquestions: int          # len(active_tasks)
    evidence_sufficient: bool      # _has_sufficient_evidence(...)
    prev_evidence_score: float     # metrics["evidence_score_final"] before round
    curr_evidence_score: float     # after round
    consecutive_rejections: int
    model_emitted_answer: bool

class StopReason(Enum):
    CONTINUE = "continue"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLATEAU = "plateau"

class AnswerVerb(Enum):
    ACCEPT = "accept"   # take the model's <answer>
    REJECT = "reject"   # send feedback, let it search more
    FORCE  = "force"    # one final answer turn, accept unconditionally

@dataclass(frozen=True)
class StopDecision:   reason: StopReason
@dataclass(frozen=True)
class AnswerDecision: verb: AnswerVerb; feedback: str = ""

class LoopController:
    def __init__(self, cfg: SearchAgentLoopConfig): ...
    def effective_search_limit(self, num_subquestions: int) -> int: ...
    def should_continue_searching(self, s: LoopSnapshot) -> StopDecision: ...
    def final_answer_decision(self, s: LoopSnapshot) -> AnswerDecision: ...
```

**Why stateless:** `run()` is already the single state owner; the Phase-2 state
machine wants one state home. A stateful controller would create a second one.
Pure decisions also unit-test with zero setup.

### Where `run()` calls it

- After each search round → `should_continue_searching(...)`; on `PLATEAU` /
  `BUDGET_EXHAUSTED`, break out of search and route to the answer path.
- At the answer-gate (`search.py:1052`) → `final_answer_decision(...)` returns
  ACCEPT / REJECT / FORCE, replacing the hand-rolled
  `require_sufficient_evidence_before_answer` + `max_answer_rejections` block.
- At every dead-end exit (no_action, format_error_limit, budget) → funnel into the
  same FORCE path instead of `break` with `final_answer=None`.

## Behavior mapping

| Behavior | Lives in | Mechanism |
|---|---|---|
| Plateau early-stop | `should_continue_searching` | `STOP(PLATEAU)` when `curr - prev < min_gain` **and** `evidence_sufficient` (floor guard — never plateau-stop into an empty answer) |
| Adaptive budget | `effective_search_limit` | `clamp(base + bonus·max(0, n_subq − 1), base, cap)` — single-subquestion questions keep today's limit |
| Graceful dead-end | `final_answer_decision` → FORCE | dead-end exits run one final answer turn instead of returning nothing |
| Smarter gating | `final_answer_decision` → REJECT | targeted per-subquestion feedback, then FORCE once `rejections ≥ cap` |

### New config knobs (defaults)

```python
evidence_plateau_min_gain: float = 0.05      # was None → now ON
plateau_requires_sufficient: bool = True     # floor guard
search_budget_per_subquestion: int = 1       # +1 round per extra subquestion
max_search_limit_cap: int = 10               # adaptive ceiling
force_answer_on_deadend: bool = True         # dead-ends emit best-effort answer
```

Decisions: **plateau = conservative** (gain<0.05, gated on sufficiency — can only
trim a spinning, already-answerable run; never cuts while evidence is climbing or
insufficient). **Adaptive budget keyed on subquestion count** (an already-tracked
explicit signal); single-subquestion questions — the common case — see zero
behavior change.

## Forced-final-answer turn

Unifies graceful-dead-end and smarter-gating.

**Triggered when** (any): budget/plateau stop with no accepted answer ·
`rejections ≥ max_answer_rejections` · no-action/format-error dead-end **while
evidence exists**.

**Behavior** — exactly one extra generation turn:
1. Append a user message instructing the model to answer now from collected
   evidence only, state uncertainty, and cite evidence labels.
2. Generate once; accept its `<answer>` **unconditionally** (gate bypassed).
3. **Bounded fallback:** if the forced turn still emits no `<answer>`, fall back to
   the last tentative `final_answer` (`search.py:1047`); if none, synthesize a
   minimal templated answer from top cited evidence. Never loops, never returns
   `None` when evidence exists.

**Guardrails:**
- Strictly one generation — a `forced_answer_attempted` flag prevents re-entry.
- Counts against `num_turns` but **not** `rounds_used` (synthesis, not search).
- **No-evidence opt-out:** if no evidence was collected at all, keep today's empty
  exit — do not fabricate.

## Metrics & reward contract

**Rule: add metrics, never mutate existing key semantics.**

| New metric | Meaning | Reward treatment |
|---|---|---|
| `forced_final_answer` | A dead-end/cap forced a salvage answer | new `forced_final_answer_penalty = -0.05` |
| `plateau_early_stop` | Loop actually stopped on plateau (vs. observability-only `early_stops`) | reuses `early_stop_bonus` |
| `effective_search_limit` / `adaptive_budget_bonus` | Adaptive budget chosen | observability only |

**Corrections to avoid mis-training:**
1. **Forced answers are mutually exclusive with `answer_when_evidence_insufficient`.**
   A forced salvage sets `forced_final_answer=1` and **not**
   `answer_when_evidence_insufficient=1`, so the model is not hit with the harsh
   `-0.2` for doing what it was told at a dead-end. Forced = `-0.05` (better than
   nothing, worse than a clean answer); voluntary-insufficient keeps `-0.2`.
2. **`search_budget_exhausted_without_answer` falls toward 0** — the intended
   reward improvement (we now answer instead of returning nothing). No change to
   its penalty; the metric simply stops firing.

**Why existing reward tests stay green:** `test_reward.py` feeds metric dicts
directly and reads keys via `.get(key, 0)`; new keys default to 0. The byte-stable
`_zeroed` preset (`reward.py:282`) gets `forced_final_answer_penalty=0.0` added so
it stays byte-stable. The accepted baseline shift lands in **loop** tests
(`test_agent_loop.py`), not reward-weight tests.

## Testing strategy

**1. `LoopController` unit tests (new — `tests/unit/test_loop_controller.py`).**
Pure decisions, zero model/tokenizer setup. Written **first (TDD)**.
- `should_continue_searching`: CONTINUE while evidence climbing · STOP(PLATEAU)
  only when `gain<0.05 AND sufficient` · no plateau-stop when insufficient ·
  STOP(BUDGET_EXHAUSTED) at `effective_limit`.
- `effective_search_limit`: 1 subq → base · n subq → clamp formula · cap respected.
- `final_answer_decision`: ACCEPT when sufficient · REJECT(+feedback) while
  `rejections < cap` · FORCE at cap · FORCE on dead-end-with-evidence.

**2. Loop integration tests (extend `test_agent_loop.py`).** Existing fake
model/server harness.
- Plateau path: stalled-evidence script → loop stops early, `rounds_used` drops,
  `plateau_early_stop=1`.
- Forced-answer path: budget/no-action dead-end with evidence → exactly one extra
  turn, `final_answer` non-None, `forced_final_answer=1`,
  `search_budget_exhausted_without_answer=0`.
- Bounded fallback: forced turn emits no `<answer>` → tentative/templated, no loop.
- No-evidence opt-out: immediate format errors → keep today's empty exit.
- Baseline updates: existing tests whose `rounds_used`/answer expectations shift
  are updated in the same commit (enumerated in the implementation plan).

**3. Reward tests (extend `test_reward.py`).** New `forced_final_answer_penalty`
pricing; `_zeroed` preset stays byte-stable; mutual-exclusivity (forced answer
priced `-0.05`, not `-0.2`).

**Verification gate:** full `pytest` green before any commit; diff the metrics dict
keys against `reward.py` / `action_eval.py` consumers to prove no existing key
changed meaning.

## Files touched

- **New:** `src/agents/components/loop_controller.py`
- **New:** `tests/unit/test_loop_controller.py`
- **Edit:** `src/agents/search.py` (wire controller into `run()`; add config knobs;
  remove inline plateau/gate/dead-end logic)
- **Edit:** `src/training/reward.py` (new `forced_final_answer_penalty`; `_zeroed`
  preset addition)
- **Edit:** `tests/unit/test_agent_loop.py`, `tests/unit/test_reward.py`

## Non-goals

- Full state-machine refactor / `run()` decomposition (Phase 2).
- Unifying the loop onto `Planner`/`SearchTool` components (Phase 2).
- Difficulty-classifier-based budgeting (subquestion count only).
- Fabricating answers when no evidence was collected.
