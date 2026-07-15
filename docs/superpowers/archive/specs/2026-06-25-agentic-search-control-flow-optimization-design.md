# Agentic Search Control Flow Optimization Design

## Objective

Borrow the useful control-flow ideas from the sampled Basic agent code and apply
them to `SearchAgentLoop` without importing `minisweagent`, changing the XML
action contract, or rewriting the agent into a new framework.

The sampled `DefaultAgent` code has a clean lifecycle:

- query and action execution are separate phases;
- limits are checked before model calls;
- repeated format errors have an explicit exit path;
- every run ends with a structured exit status;
- trajectory serialization is centralized.

`SearchAgentLoop` already has richer search behavior, evidence gating, fetches,
subquestion tracking, and reward metrics. The optimization should keep those
behaviors and make the loop easier to reason about by adding explicit lifecycle
outcomes and small internal phase helpers.

## Assumptions

- The sample code is reference material. It should not remain appended to
  `examples/run_agentic_search.py` as production CLI code because it introduces
  new imports and dependencies after `asyncio.run(main())`.
- Public agent APIs stay stable. `SearchAgentLoop.run(...)` still returns
  `AgentLoopOutput`.
- The XML action tags stay unchanged.
- Evidence sufficiency and answer rejection rules stay unchanged.
- This pass optimizes control flow observability and maintainability, not model
  policy, reward weights, or retrieval quality.

## Proposed Approach

Use a native step-outcome control-flow pass.

1. Add explicit loop exit status.
   - Track why a run ended: `answered`, `max_turns`, `search_limit`,
     `format_error_limit`, `no_action_exit`, or `exception`.
   - Represent status with numeric counters such as `exit_answered`,
     `exit_max_turns`, and `exit_format_error_limit` because current metrics are
     treated as `dict[str, float]` throughout training/reward code.

2. Add no-action/format-error accounting.
   - Reuse existing no-action feedback behavior.
   - Add `format_error_turns` and `max_consecutive_format_errors` config.
   - When the model repeatedly emits no recognized action, stop with
     `format_error_limit` instead of relying only on max turns.

3. Extract small private phases from `SearchAgentLoop.run`.
   - Keep behavior unchanged, but isolate responsibilities:
     - generation and parsing;
     - subquestion registration;
     - answer gating;
     - observation assembly;
     - final metrics.
   - Avoid a full state-machine rewrite in this PR.

4. Handle sampled code.
   - Remove the appended Basic agent sample from `examples/run_agentic_search.py`
     after the ideas have been captured in this spec and implemented natively.

## Component Impact

### SearchAgentLoopConfig

Add one conservative config field:

- `max_consecutive_format_errors: int = 3`

This mirrors the sample's repeated-format-error limit. Existing behavior already
has answer rejection limits; this adds the same guard for malformed/no-action
model turns.

### SearchAgentLoop

Keep the existing public method and return type. Internally:

- count no-action turns in `format_error_turns`;
- track consecutive no-action turns;
- set a terminal exit counter before returning;
- extract private helpers only where they remove real complexity from `run()`.

### AgentLoopOutput

No dataclass change is required. Exit status can be represented in the existing
metrics dictionary with numeric counters, avoiding any reward-code disturbance.

## Metrics

Add these metrics:

- `format_error_turns`: total no-action turns that required feedback or exit.
- `exit_answered`: `1.0` when the run ended with an accepted answer.
- `exit_max_turns`: `1.0` when the turn loop exhausted `max_turns`.
- `exit_search_limit`: `1.0` when search budget was exhausted without an answer.
- `exit_format_error_limit`: `1.0` when repeated no-action turns hit the new
  limit.
- `exit_no_action`: `1.0` when the loop exits on a no-action turn after evidence
  is already sufficient or rejection limits are exhausted.

These are observational by default. Reward code does not need to consume them in
this pass.

## Testing Strategy

Add unit tests in `tests/unit/test_agent_loop.py`:

- normal answer sets `exit_answered`;
- repeated no-action turns increment `format_error_turns` and eventually set
  `exit_format_error_limit`;
- max-turn exhaustion sets `exit_max_turns`;
- search-limit exhaustion still sets the existing
  `search_budget_exhausted_without_answer` and also sets `exit_search_limit`;
- existing evidence and answer-gating tests continue to pass.

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "exit or format_error or search_limit" -v
pytest tests/unit/test_agent_loop.py -v
```

## Out of Scope

- Importing or depending on `minisweagent`.
- Changing the XML action vocabulary.
- Changing reward weights.
- Changing retrieval, reranking, evidence scoring, or citation behavior.
- Replacing `SearchAgentLoop` with a full state-machine architecture.

## Success Criteria

- `SearchAgentLoop` exposes clear terminal outcome metrics.
- Repeated malformed/no-action model output has a deterministic stop condition.
- The sampled Basic agent code is no longer appended to the CLI execution file.
- Public APIs remain backward compatible.
- Focused and full `test_agent_loop.py` suites pass.
