# Promotion readiness: what would have to be true, and how it rolls back

## Status

Not started. **This spec does not promote the router**, and does not argue that
it should be promoted. It defines the checklist, the rollout, and the rollback,
so that when the decision is taken it is taken against stated criteria rather
than against whichever number is quoted that day.

## Problem

`AGENTIC_SEARCH_INTENT_INDEX_PATH` is unset by default, so learned routing is
dark and every request falls through the LLM/rule cascade. Six PRs have improved
the artifact behind that flag, and each has ended with some version of
"promotion is a separate change, reviewed on its own terms."

Nobody has written down what that review consists of. The current evidence is
genuinely mixed in a way a single accuracy number hides:

**For:**
- served accuracy `1.0000` on the test slice — 58 answered, zero misroutes
- argmax `0.8108`, clearing the pre-registered `0.80` band
- out-of-scope AUC `0.8720`, its best measured value

**Against, or at least unresolved:**
- the bar is cleared by **two queries** on a 111-query slice
- out-of-scope AUC is still the one measure MiniLM wins on matched anchors
  (`0.8863`), and the `0.90` bar #512 set for itself has never been met
- coverage is `0.523` — the router defers **half** of all traffic to the LLM
  classifier, which is the cost objection nobody has priced
- p95 routing latency `12.20ms` is added to every auto-routed request, on top of
  a ~2s one-time encoder load paid by the first caller

## What this spec must produce

**1. A promotion checklist**, fixed in advance, covering at minimum: which
metrics must hold and at what values, what the coverage/cost tradeoff must look
like, and what evidence is required about *production* traffic rather than the
eval sets. #521 made abstention observable in production specifically so this
question could be answered with data; that data has not been collected yet.

**2. A staged rollout.** The flag is currently all-or-nothing per process. A
percentage rollout, or a shadow mode that scores every request and records what
it *would* have routed without acting on it, converts promotion from a bet into
a measurement. Shadow mode is the stronger option and reuses the telemetry
already added.

**3. A rollback trigger and procedure.** Unsetting the variable is the mechanism,
but the trigger needs stating — and note the operational trap already documented:
**the index is cached by resolved path and never invalidated, and a failed load
is cached too.** So rollback requires a process restart, and enabling requires
the index to exist *before* the process starts. Any rollout plan that ignores
that will produce a confusing partial state.

**4. A decision, recorded either way.** "Not yet, because X" is a perfectly good
output and is more useful than a seventh PR that improves the artifact without
addressing whether it ships.

## Acceptance

- A written checklist with pre-registered thresholds, in
  `docs/training-and-evaluation.md`.
- A shadow-mode or staged-rollout mechanism, or an explicit recorded decision
  that all-or-nothing is acceptable and why.
- Rollback documented including the cache-restart requirement.
- If promoting: the activation steps are a checklist someone else can execute.
- If not: the specific unmet condition is named, so the next attempt knows what
  it is aiming at.

## Out of scope

Any further tuning of the router. If the checklist is not met, the answer is to
record that — not to tune until it is, which would fit the artifact to its own
promotion criteria.

## Dependency worth noting

[The wider eval instrument](2026-08-14-intent-wider-eval-instrument-design.md)
would make this decision much better founded. A two-query margin is not a sound
basis for a promotion checklist, and doing that work first is the cheaper
ordering.
