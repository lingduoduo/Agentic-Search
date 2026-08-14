# Plan: a judge that reads the gold answer

Spec: `docs/superpowers/specs/2026-08-14-grpo-real-judge-design.md`

## Two decisions taken before implementing

Both were put to the owner rather than assumed, because the spec identified them
as direction rather than mechanics:

1. **LLM-as-judge with a deterministic offline fallback**, not a pure-offline
   matcher and not an LLM with no fallback.
2. **One documented default, the others opt-in** — resolving the trap where two
   judges with incompatible contracts were selected by which example script ran.

## What was actually wrong

`BatchJudgeFn` is `Callable[[list[str], list[str]], list[float]]` — the seam has
*always* passed the ground truths. `SimulatedPreferenceJudge.as_batch_judge_fn`
simply discarded them, and `make_judge_fn` in the trainer example discarded them
again with a parameter named `_ground_truth`.

So this was never a missing interface. It was a judge, and an adapter, both
throwing away the argument that makes a correctness signal possible.

## Steps

### 1. `GoldAgreementJudge` — deterministic, reference-based

Exact match → `1.0`, containment → `0.7`, else scaled token-F1. Reuses the
existing `normalize_answer_text` and `token_f1_score`.

→ verify: a terse-correct answer outscores a fluent-wrong one, and the
*previous* judge gets that pair backwards. Both asserted in one test so the
regression cannot return quietly.

### 2. `LLMJudge` — LLM over `(answer, gold)`, fallback per item

→ verify: with `llm=None` it equals `GoldAgreementJudge` exactly and makes zero
calls, which is what keeps the no-network smoke path working.

### 3. Fail loudly, then degrade narrowly

`parse_judge_score` raises on empty, non-numeric, or out-of-range replies.
`LLMJudge` catches, falls back **for that item only**, and counts it.

→ verify: one bad reply in a batch does not flatten the group, the good rollouts
keep their LLM scores, and `parse_failures` records it.

### 4. Cache by `(answer, gold)`

→ verify: four identical scorings make one call; changing the gold makes a second.

### 5. Make it the default; keep the old judge reachable

`--judge gold|llm|simulated`, defaulting to `gold`.

→ verify: full suite green.

## The subtle failure this is mostly defending against

Every guard here — graded partial credit, raising on unparseable replies,
`is_degenerate_group` — protects the same thing: **GRPO normalises advantages
within a rollout group, so any judge that returns the same score for every
rollout of a prompt produces all-zero advantages.** That prompt then contributes
nothing to the update while the step still logs as if it had.

A binary judge does this on every prompt the policy gets wrong. A judge that
defaults unparseable replies to a constant does it whenever the provider
misbehaves. Neither surfaces as an error; both surface as "training ran and the
model did not move", which is expensive to diagnose after the fact.

## Verification

Full suite: **3188 passed**, 3 deselected — the hardware-sensitive latency bar
and two process-timing MCP tests, all three confirmed to pass in isolation both
with and without this change.
