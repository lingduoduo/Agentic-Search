# Route clarification and guess-site honesty — design

**Date:** 2026-08-12
**Status:** Approved
**Depends on:** PR #507 (route telemetry, calibration, out-of-scope probes)

## Problem

The intent router always returns one of `chat`, `search`, or `tool`, even when
nothing in the cascade had a signal. Two sites guess and present the guess as a
decision:

- `classify_route` defaults to `chat` when the LLM returns empty or unparseable
  text.
- `_rule_based_route` returns `chat` as its "no dominant signal" default.

The reference behavior for an ambiguous request such as "Review the vendor
renewal terms" is an LLM fallback *or a clarification*. No clarification path
exists anywhere in the codebase: `rg -ni "clarif|disambiguat|ask_the_user"` over
the web server and agents returns nothing. The request silently becomes a chat
answer, and the user is never told the router was guessing.

Two smaller defects compound the same theme of unclear signals:

- The mechanism labels are a trap. `regex` means "high-confidence rules
  decided"; `rule_based` means "everything else failed". Same word, opposite
  ends of the cascade.
- The `tool_precision_minimum` promotion gate reads cascade-wide tool precision
  rather than the model's own tool routes, so deterministic routes dilute the
  measurement the gate exists to make.

## Goal

Let the router say "I don't know" and ask, without changing any path that
already had a signal, and without widening the three-label intent vocabulary.

## Scope

### In scope

- A clarification outcome at both guess-sites.
- A request field that re-enters the auto dispatcher with a chosen route.
- Mechanism renaming for unambiguous traces.
- A model-scoped tool-precision promotion gate.

### Out of scope

- Routing `/chat`, `/search`, and `/tools` through the Assistant. Those are
  separate endpoint stacks; consolidating them is its own project.
- Changing what happens when a `tool` route has no local model. It degrades to
  chat and records `route_degraded`, which stays as is.
- LLM-generated clarification text.
- Any change to `INTENT_LABELS`, the checkpoint contract, or the served model.

## Architecture

The cascade keeps its order. Every branch that had a signal is untouched:

```text
explicit source                  -> search                unchanged
rules (regex)                    -> strategy              unchanged
intent model, conf >= threshold  -> strategy              unchanged
LLM classifier, usable label     -> strategy              unchanged
LLM classifier, empty/unparsed   -> CLARIFY               was: silent chat
LLM classifier, raised           -> fall through          unchanged
heuristic, cue matched           -> strategy              unchanged
heuristic, no signal             -> CLARIFY               was: default chat
```

An empty query continues to return `chat` rather than clarifying. It is
degenerate input the endpoint already handles, and asking a user to
disambiguate an empty request is not useful.

## Components

### Clarification is not a fourth intent

`RouteStrategy` is not only a routing enum. `ml_intent._ROUTE_VALUES` derives
from it to validate model predictions, and `intent_routing._LABEL_BY_VALUE`
derives the LLM classifier's label map from it. Adding a `CLARIFY` member would
let a checkpoint predicting `"clarify"` pass validation and let a stray
"clarify" in an LLM completion match a label its prompt never offers.

Clarification is therefore the *absence* of an intent, represented outside the
enum:

```python
@dataclass(frozen=True)
class ClarificationOption:
    route: str      # "chat" | "search" | "tool"
    label: str


@dataclass(frozen=True)
class Clarification:
    question: str
    options: tuple[ClarificationOption, ...]


@dataclass(frozen=True)
class RouteDecision:
    strategy: RouteStrategy | None
    clarification: Clarification | None
```

Exactly one of `strategy` and `clarification` is set.

### Entry points

`route_request(query, *, llm, explicit_source, settings=None, telemetry=None)
-> RouteDecision` becomes the entry point and holds the cascade.

`route_query` keeps its exact signature and return type, implemented as
`decision.strategy or _rule_based_route(query)`. Because clarification fires
precisely where the heuristic would have returned its no-signal default, that
wrapper is behavior-identical to today, so existing callers and their tests are
unaffected. `app.py` is the single production caller and moves to
`route_request`.

### Supporting changes

`_rule_based_route_or_none(query) -> RouteStrategy | None` returns `None` for
the no-signal case; `_rule_based_route` becomes a wrapper returning
`... or RouteStrategy.CHAT`.

`classify_route` returns `tuple[RouteStrategy | None, dict]`, yielding `None`
for the `empty` and `unexpected` raw labels instead of defaulting to chat. The
detail payload is unchanged and still carries no prompt or completion text.

The clarification question and its three options are static. This fires
precisely when no usable LLM produced a label, so generating the text with an
LLM is either impossible or an immediate retry of something that just failed to
emit one word of three.

## Data flow

```text
POST /api/agent {query}
  -> intent="clarify", clarification{question, options}, no agent runs

POST /api/agent {query, route:"search"}
  -> router skipped -> auto dispatcher -> search agent
```

The follow-up uses a new optional `route` field accepting exactly `chat`,
`search`, or `tool`, rather than reusing the six legacy `mode` values. The
legacy modes have different semantics from the auto-routed branches —
`hybrid_search` is not what the auto search branch runs — so mapping the three
choices onto them would dispatch a different agent than the one the user was
offered. The `route` field short-circuits the router and enters the same auto
dispatcher, preserving its degradation behavior.

Both `/api/agent` and `/api/agent/stream` carry the clarification. The response
gains an optional `clarification` field, populated only when
`intent == "clarify"`; `answer` holds the question text and citations and
documents are empty.

### Telemetry

`hook_metadata.route_mechanism` is `"clarify"` on the question and
`"user_selected"` on the follow-up. Because the query is already persisted with
the session, a user's selection becomes a ground-truth label joinable to it,
which feeds the route telemetry into a corrected training set. Nothing new
about the request is logged.

## Mechanism renaming

| Today | New | Meaning |
|---|---|---|
| `regex` | `rules` | Deterministic high-precision cues decided |
| `rule_based` | `heuristic_default` | Nothing else worked; heuristic guess |
| — | `clarify` | No signal; the user was asked |
| — | `user_selected` | The user chose the route |

`explicit_source`, `model`, and `classifier` are unchanged. Only the emitted
mechanism labels change; function names such as `_regex_route` and
`_rule_based_route` keep their current names, since renaming them would touch
call sites and patch targets across the test suite for no behavioral gain.
These strings appear in capture stages, `hook_metadata.route_mechanism`, and the
docs. No persisted schema constrains them.

## Model-scoped tool precision

`evaluate_intent_predictions` already separates `covered` (model-decided
records at or above threshold) from the full record set. The report gains
`model_tool_precision`, computed over covered records only, and
`tool_precision_minimum` reads it. Cascade-wide `per_label_metrics` stays in the
report for context.

When no covered record predicts `tool`, `model_tool_precision` is `None` and
the gate fails rather than reading scikit-learn's `0.0` for an empty prediction
set. Unmeasured is not evidence of safety, consistent with the out-of-scope
gate.

This is offline only. No runtime behavior changes.

## Configuration

`AGENTIC_SEARCH_ROUTE_CLARIFICATION` (default `true`) enables the clarification
outcome. When disabled, `route_request` returns the legacy heuristic strategy at
both guess-sites, which is today's exact behavior. The default is on
deliberately: an off-by-default router change would ship dark, the failure this
work exists to correct.

## Error handling

- `Clarification` is a frozen dataclass built from constants and performs no
  I/O, so constructing it cannot fail.
- A clarification never runs an agent, so no downstream failure can occur on
  that path.
- An unrecognized `route` value raises `HTTPException(422)` listing the valid
  values, matching how `_normalize_agent_mode` rejects an unrecognized `mode`.
- Explicit modes, explicit sources, deterministic rules, and confident model
  predictions never clarify.

## Verification

### Unit tests

- Heuristic no-signal clarifies; each matched cue (tool, search, bare lookup)
  routes unchanged.
- Unusable LLM output clarifies; a usable label routes unchanged.
- A raised LLM exception falls through to the heuristic, which may clarify.
- Deterministic rules and confident model predictions never clarify.
- An empty query returns `chat`.
- `route_query` remains behavior-identical across all of the above.
- Disabling clarification restores the legacy strategy at both guess-sites.
- Mechanism strings are the renamed values.

### Integration tests

- An uncertain query through `/api/agent` returns `intent="clarify"` and
  invokes none of the three runners.
- The follow-up with `route:"search"` reaches
  `_run_search_direct_or_escalate`; `chat` reaches `_run_agentic_rag`; `tool`
  reaches `_run_tool_agent`.
- The streaming endpoint carries the same clarification.
- An explicit `mode` still bypasses the router entirely.

### Promotion-gate tests

- A candidate whose model-only tool precision is `0.0` fails
  `tool_precision_minimum` even when regex-decided routes lift cascade
  precision above the limit. This is the demonstrated dilution case: forty
  correct deterministic tool routes plus two false model tool routes yield
  0.9524 cascade precision and pass today.
- A covered set with no `tool` prediction reports `None` and fails the gate.

## Success criteria

- An ambiguous request asks the user instead of silently answering as chat.
- Every request that today reaches a signal keeps its exact route and agent.
- The three-label vocabulary is unchanged, and no checkpoint or classifier can
  produce a clarification.
- A trace names the deciding mechanism unambiguously.
- The tool-precision gate measures the model's own action routes.

## Risks and mitigations

- **Clarification fatigue:** the trigger is narrow by construction — only
  genuine coin-flips reach it. If it proves noisy in practice, the operator
  switch disables it without a deploy.
- **A round trip costs latency:** the alternative is running the wrong agent,
  which costs more. The question itself runs no agent and no LLM.
- **Renaming breaks a consumer:** the mechanism strings are not a persisted
  schema; the rename is covered by tests asserting the new values.
