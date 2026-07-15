# Generated Context Pack

# Agent Control Flow Consolidation

## Sources

- [Specification: 2026-06-27-agent-control-flow-consolidation-design.md](../specs/2026-06-27-agent-control-flow-consolidation-design.md)
- [Plan: 2026-06-27-agent-control-flow-consolidation.md](../plans/2026-06-27-agent-control-flow-consolidation.md)

## Specification Context

### Goal

Make the existing workflow easier to reason about, test, and extend. The existing `AgentState` becomes the single mutable state object used by the agent loop and its five existing components:

- `Planner`
- `SearchTool`
- `RerankerTool`
- `EvidenceJudge`
- `AnswerGenerator`

Remove the duplicate `SearchAgentState`, route work through the existing components, and make `LoopController` the explicit home of continuation and answer policy. Do not introduce another state class, compatibility wrapper, or alias. Preserve current search-loop behavior, output shape, metrics, XML actions, retries, caching, and answer gating.

### Architecture

`SearchAgentLoop` remains the public loop implementation and lifecycle owner. It creates one `AgentState` per run and delegates decisions and mutations to the existing components.

`LoopController` remains stateless. It receives a `LoopSnapshot` derived from `AgentState` plus transient control counters such as active subquestion count and consecutive answer rejections. Those counters are not added to `AgentState` because the requested canonical search state has only the six existing search concerns.

…

### Decision ownership

| Decision | Owner | Inputs | Outcomes |
|---|---|---|---|
| What action did the model request? | Planner | generated XML, previous queries | search, rerank, fetch, answer, or format recovery |
| Is another search allowed? | Loop Controller | search rounds, effective budget, evidence gain and sufficiency | continue, budget exhausted, or plateau |
| Which backend executes the search? | Search Tool | planned retriever and query batch | vector DB, web, or configured fallback |
| Is evidence sufficient? | Evidence Judge | accumulated search contexts | score plus boolean sufficiency verdict |

…

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

## Implementation Plan Context

### Task 1: Add canonical search fields to the existing `AgentState`

**Files:**
- Modify: `src/agents/state.py`
- Modify: `tests/unit/test_agent_state.py`
- Modify: `tests/unit/test_state_models.py`

**Interfaces:**
- Produces: `AgentState.record_search_round(queries: list[str], docs: list[SearchResult]) -> None`
- Produces: unchanged `record_rerank`, `set_evidence`, `set_citations`, and `to_dict` behavior on `AgentState`
- Preserves temporarily: `SearchAgentState` until component callers migrate in Task 2

- [ ] **Step 1: Rewrite state tests to instantiate the existing `AgentState`**

Use one helper so every test supplies the pre-existing required orchestration fields:

…

### Task 2: Migrate all five components to `AgentState`

**Files:**
- Modify: `src/agents/components/planner.py`
- Modify: `src/agents/components/search_tool.py`
- Modify: `src/agents/components/reranker_tool.py`
- Modify: `src/agents/components/evidence_judge.py`
- Modify: `src/agents/components/answer_generator.py`
- Modify: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: `AgentState` and its Task 1 mutation methods
- Produces: `Planner.decide(text: str, state: AgentState) -> PlannerDecision`
- Produces: `SearchTool.run_round(state, queries, retriever) -> list[list[SearchResult]]`

- [ ] **Step 1: Change component tests to use the Task 1 state helper**

…

### Task 3: Make Planner the loop's parsing entry point

**Files:**
- Modify: `src/agents/components/planner.py`
- Modify: `src/agents/search.py`
- Modify: `tests/unit/test_components.py`

**Interfaces:**
- Produces: `Planner.parse_actions(text: str, action_tags: Sequence[str]) -> list[tuple[str, str]]`
- Produces: `Planner.round_retriever(text: str, search_tags: Sequence[str]) -> Retriever`
- Produces: `Planner.round_rerank(text: str, search_tags: Sequence[str]) -> bool`

- [ ] **Step 1: Add parser parity tests before moving loop helpers**

- [ ] **Step 2: Run the parity test and confirm failure**

Run: `pytest tests/unit/test_components.py::test_planner_parses_complete_mixed_turn -v`

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
