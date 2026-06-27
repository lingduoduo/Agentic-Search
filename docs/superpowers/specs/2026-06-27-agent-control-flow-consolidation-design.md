# Agent Control-Flow Consolidation Design

**Date:** 2026-06-27
**Status:** Approved for implementation planning
**Scope:** Consolidate the existing search-agent state and component control flow without changing the public XML protocol or observable loop behavior.

## Goal

Make the existing `AgentState` the single mutable state object used by the agent loop and its five existing components:

- `Planner`
- `SearchTool`
- `RerankerTool`
- `EvidenceJudge`
- `AnswerGenerator`

Remove the duplicate `SearchAgentState`. Do not introduce another state class, compatibility wrapper, or alias. Preserve current search-loop behavior, output shape, metrics, XML actions, retries, caching, and answer gating.

## Current Problem

The repository already contains the five components and a six-field `SearchAgentState`, but `SearchAgentLoop` still owns equivalent data in local variables, `AgentContext`, and `metrics`. Consequently, the component layer is tested mostly in isolation and is not the authoritative execution path.

The duplicated state is:

- question
- previous queries / `executed_queries`
- retrieved documents / `AgentContext`
- evidence score / metrics
- search rounds / `rounds_used`
- citations / final answer post-processing

There is also a separate orchestration-level `AgentState`, exported publicly from `src`. Maintaining both state dataclasses creates two competing meanings for agent state.

## Chosen Approach

Extend the existing orchestration `AgentState` with the six existing search fields and transition helpers, then delete `SearchAgentState`.

The existing orchestration fields remain unchanged for compatibility. The search loop constructs `AgentState` with its request ID and a `UserRequest` derived from the incoming messages. The canonical search fields are:

```python
question: str
previous_queries: list[str]
retrieved_docs: list[SearchResult]
evidence_score: float
search_rounds: int
citations: list[Citation]
```

The existing helpers move to `AgentState`:

- record a completed search round
- replace the working document order after reranking
- clamp and set evidence score
- replace structured citations

`search_rounds` retains the current `SearchAgentLoop` meaning: completed search rounds, not the number of parallel queries in a round. Duplicate or blocked queries do not increment it.

## Alternatives Rejected

### Replace `AgentState` with the six-field shape

This would remove routing, planning, memory, tool results, tracing, and response fields used by existing callers. It creates an unnecessary public API break.

### Keep `SearchAgentState` as an alias or adapter

This reduces migration work but retains two names and two conceptual APIs. It does not genuinely consolidate state.

### Keep loop locals and only rename types

This is cosmetic. The loop and components would continue to disagree about the source of truth.

## Architecture

`SearchAgentLoop` remains the public loop implementation and lifecycle owner. It creates one `AgentState` per run and delegates decisions and mutations to the existing components.

```text
model generation
      |
      v
Planner ---------> typed action(s)
      |                  |
      |          +-------+--------+
      |          |       |        |
      v          v       v        v
 SearchTool  Reranker  Evidence  AnswerGenerator
      |        Tool      Judge         |
      +----------+---------+-----------+
                           |
                           v
                     AgentState
                           |
                           v
                    LoopController
                 (pure policy decisions)
```

`LoopController` remains stateless. It receives a `LoopSnapshot` derived from `AgentState` plus transient control counters such as active subquestion count and consecutive answer rejections. Those counters are not added to `AgentState` because the requested canonical search state has only the six existing search concerns.

`AgentContext` remains the compatibility representation returned in `AgentLoopOutput.context` and the source of formatted evidence labels. It is updated alongside `AgentState` at the component boundary, but it no longer owns counters, evidence score, query history, or citations.

## Component Responsibilities

### Planner

Parse the existing XML generation into typed actions. It reads `AgentState.previous_queries` for duplicate detection. The loop retains support for the full current vocabulary: decision, subquestions, single and parallel searches, fetch, rerank attributes, and answer.

The current standalone planner understands only a subset of that vocabulary. Consolidation therefore extends it to produce a turn-level plan that represents all recognized actions without changing precedence or mixed-action behavior.

### Search Tool

Own retriever selection, query execution, per-run caching, web-to-vector fallback, and state recording. It accepts the allowed queries for one round, executes independent queries concurrently as today, returns `SearchContext` objects, and records the round exactly once in `AgentState`.

Result deduplication and citation labels retain their current ordering. Repeated and overflow queries are filtered before execution and do not mutate the canonical state.

### Reranker Tool

Rerank only when requested by the existing `rerank="true"` action and when enough documents exist. It updates `AgentState.retrieved_docs` without incrementing `search_rounds`. Existing candidate limits, fallback behavior, and rerank metrics remain intact.

### Evidence Judge

Evaluate the same `SearchContext` collection with the current `SearchResultEvaluator`. It writes the continuous score to `AgentState.evidence_score` and returns the existing boolean sufficiency verdict used by answer gating and `LoopController`.

Metrics derive from the verdict and state; metrics are observability output rather than a second state store.

### Answer Generator

Accept the model's answer text, resolve only valid evidence markers, update `AgentState.citations`, and return the same final answer text. `AgentLoopOutput` remains unchanged; existing string citation metrics continue to be emitted from the structured citations.

### Loop Controller

Remain the single policy location for search continuation and final-answer decisions. Budget checks read `AgentState.search_rounds`; plateau checks read `AgentState.evidence_score`; answer decisions consume the Evidence Judge verdict. Forced-answer generation remains an effect performed by the loop after a controller decision.

## Data Flow

1. The loop derives `question` from the latest user message and creates one `AgentState`.
2. The model generates the existing XML response.
3. `Planner` converts the response into typed turn actions using state query history.
4. The loop applies non-I/O declarations such as subquestions.
5. `LoopController` and the current budget policy determine which requested searches are allowed.
6. `SearchTool` executes the allowed round and records query history, documents, and one search round.
7. `RerankerTool` optionally reorders the current documents.
8. `EvidenceJudge` updates the evidence score and sufficiency verdict.
9. `LoopController` decides whether to continue, reject, accept, or force an answer.
10. `AnswerGenerator` resolves citations and updates state.
11. The loop derives metrics and the unchanged `AgentLoopOutput` from state and compatibility context.

## Compatibility and Error Handling

- Preserve `SearchAgentLoop.run(...)` and `AgentLoopOutput` signatures.
- Preserve the current XML protocol and mixed-action behavior.
- Preserve vector and web retriever configuration and web fallback.
- Preserve retries, timeouts, cache behavior, result deduplication, and client cleanup.
- Preserve graceful forced answers and evidence-gated answer rejection.
- Retriever or reranker failure follows existing degradation behavior; state mutates only after a completed operation.
- Invalid model output follows the existing format-error and forced-answer path.
- `AgentState.to_dict()` continues to serialize orchestration fields and now also serializes the six search fields.

No new dependency, endpoint, configuration setting, or model action is introduced.

## Migration Sequence

1. Merge the six fields and helpers into `AgentState`; migrate state tests and component type annotations.
2. Extend Planner to represent the loop's complete existing action vocabulary.
3. Route search, rerank, evidence, and answer operations through the existing components.
4. Replace duplicate loop locals with reads from `AgentState` one concern at a time.
5. Remove `SearchAgentState` and its export after all references are migrated.
6. Keep `AgentContext` only where required for evidence formatting, citations, output compatibility, and reward consumers.

This ordering keeps each change reviewable and allows behavior-regression tests to pin the loop throughout the refactor.

## Testing

### State tests

- Existing orchestration construction and serialization remain valid.
- The six search fields have independent default containers.
- Search rounds increment once per executed round.
- Previous queries remain ordered and deduplicated.
- Evidence score is clamped to `[0, 1]`.
- Reranking and citation replacement do not affect search-round count.

### Component tests

- All five components accept `AgentState`; no `SearchAgentState` reference remains.
- Planner covers every currently supported XML action and mixed-action precedence.
- Search Tool covers parallel queries, caching, fallback, repeats, overflow, and failures.
- Reranker, Evidence Judge, and Answer Generator mutate only their owned fields.

### Loop regression tests

- Direct answer, single search, parallel search, web search, rerank, fetch, repeated query, overflow, plateau, answer rejection, and forced answer produce the same externally visible behavior.
- Metrics remain equal for fixed mocked trajectories.
- Returned `AgentContext`, citation labels, action trace, and response masks remain unchanged.

### Verification commands

```bash
pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py -v
pytest tests/unit/test_components.py -v
pytest tests/unit/test_agent_loop.py -v
pytest tests/unit/test_reward.py -v
ruff check src/agents tests/unit/test_agent_state.py tests/unit/test_components.py
ruff format --check src/agents tests/unit/test_agent_state.py tests/unit/test_components.py
```

Run the full default `pytest` suite before completion.

## Success Criteria

1. `SearchAgentState` no longer exists or appears in source and tests.
2. The existing exported `AgentState` contains the six canonical search fields alongside its existing orchestration fields.
3. One `AgentState` instance is threaded through Planner, Search Tool, Reranker Tool, Evidence Judge, Answer Generator, and loop control.
4. `SearchAgentLoop` no longer maintains duplicate query, document, evidence-score, search-round, or citation state in locals or metrics.
5. Existing public APIs, XML behavior, metrics, and `AgentLoopOutput` remain compatible.
6. Focused and full test suites pass, and lint/format checks are clean.

## Out of Scope

- New agent actions or XML tags
- New state or controller classes
- Changes to retrieval endpoints or response contracts
- Changes to GRPO rewards or training policy
- Removal of `AgentContext` from the public output
- Refactoring unrelated orchestration or graph-agent state types
