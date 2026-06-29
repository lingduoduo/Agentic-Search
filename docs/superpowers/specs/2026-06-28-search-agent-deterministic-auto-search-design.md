# Search-Agent Deterministic Auto-Search — Design

Status: shipped (consolidated in PR #347, alongside the router and the dispatch
consolidation). This doc covers the **auto-search** piece.
Date: 2026-06-28 (scope corrected 2026-06-29)

## Goal

Guarantee `SearchAgentLoop` runs **at least one retrieval round** even when the
policy model never emits a recognized `<search>` tag — so retrieval is not
entirely LLM-gated.

## Problem

`SearchAgentLoop.run` only retrieves when the model emits a parseable `<search>`
tag. A weak model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) that never does dead-ends on
format recovery with **zero retrieved documents** — retrieval never runs.

## Approach

Add `SearchAgentLoopConfig.auto_search_on_deadend` (default `True`). The trigger
fires **at the format-error dead-end**, not on the first no-action turn:

> when a turn yields no recognized action, **no search has run**
> (`state.search_rounds == 0`), and this turn would hit the consecutive-format-error
> limit (`consecutive_format_errors + 1 >= max_consecutive_format_errors`),
> synthesize `[(search_tag, question)]` on the user's question and feed it through
> the existing search path; record an `auto_search` control-flow event.

Firing at the dead-end (rather than turn 1) preserves the existing legitimate
behavior where an early no-action turn re-prompts for a decision. Once any search
has run, the fallback never fires again. RL rollouts set the flag `False` to
preserve dead-end penalties during GRPO training.

## Scope: what this guarantees (and what it does not)

It guarantees only that **retrieval fires at least once** — *not* a non-empty
answer. The final answer still flows through `_force_final_answer`, which requires
an `<answer>` tag and otherwise returns `None`: the **no-fabricate invariant** (a
model refusal is never passed off as an answer; see
`test_forced_turn_emitting_no_answer_returns_none`). Therefore:

- A model that emits `<answer>` when prompted with the retrieved evidence now
  produces a *grounded* answer instead of dead-ending empty (realistic weak model;
  covered by `test_deadend_forces_answer_from_evidence`).
- A genuinely tag-less model (no tag on any turn, including the forced turn) still
  returns `None` — but now with evidence retrieved rather than none. The honest
  win is "retrieval always runs", not "the answer is always non-empty".

## Tests

- `test_search_agent_loop_auto_searches_when_model_emits_no_action` — genuinely
  tag-less model: asserts `search_rounds == 1` + an `auto_search` event +
  `final_answer is None` (honest guarantee, no fabrication).
- `test_search_agent_loop_auto_search_disabled_preserves_format_recovery` — flag
  off ⇒ retrieval never runs.
- Legacy `…stops_after_repeated_no_action_turns` and
  `deadend_with_no_evidence_does_not_fabricate` pin the flag off to keep
  validating the format-error-stop / no-fabricate machinery.

## Out of scope

- Changing `_force_final_answer` / the no-fabricate invariant.
- Wiring the M10 router into the auto-search retriever (PR B).
