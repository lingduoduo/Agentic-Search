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

### Decision ownership

| Decision | Owner | Inputs | Outcomes |
|---|---|---|---|
| What action did the model request? | Planner | generated XML, previous queries | search, rerank, fetch, answer, or format recovery |
| Is another search allowed? | Loop Controller | search rounds, effective budget, evidence gain and sufficiency | continue, budget exhausted, or plateau |
| Which backend executes the search? | Search Tool | planned retriever and query batch | vector DB, web, or configured fallback |
| Is evidence sufficient? | Evidence Judge | accumulated search contexts | score plus boolean sufficiency verdict |
| Can the candidate answer finish the run? | Loop Controller | evidence verdict and rejection count | accept, reject, or force |
| Which citations are valid? | Answer Generator | answer text and compatibility context | answer plus structured citations |

This table is the control-flow contract. New behavior should be added to the owning component instead of introducing another branch in `SearchAgentLoop.run()`.

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

### Out of Scope

- New agent actions or XML tags
- New state or controller classes
- Changes to retrieval endpoints or response contracts
- Changes to GRPO rewards or training policy
- Removal of `AgentContext` from the public output
- Refactoring unrelated orchestration or graph-agent state types

## Implementation Plan Context

### Global Constraints

- Do not introduce another state class, compatibility wrapper, or alias.
- Preserve `SearchAgentLoop.run(...)`, `AgentLoopOutput`, the XML protocol, mixed-action behavior, metrics, retries, caching, result ordering, answer gating, and forced answers.
- `search_rounds` counts completed search rounds; parallel queries in one round increment it once.
- Keep existing `AgentState` orchestration fields and construction compatible.
- Add no dependency, endpoint, configuration setting, or model action.
- Keep the workflow fully automated; do not add human approval, pause/resume state, or user-response handling.
- Follow TDD for every behavior change and commit after each task passes its focused tests.

---

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

```python
from src.agents.state import AgentState, Citation, UserRequest


def _state(question: str = "q") -> AgentState:
    return AgentState(
        request_id="req-1",
        user_request=UserRequest(user_id="u1", channel="test", message=question),
        question=question,
    )


def test_record_search_round_counts_parallel_queries_once() -> None:
    state = _state()
    state.record_search_round(["first", "second"], [_doc("a"), _doc("b")])
    assert state.previous_queries == ["first", "second"]
    assert [doc.title for doc in state.retrieved_docs] == ["a", "b"]
    assert state.search_rounds == 1
```

Retain the existing clamp, rerank, citations, independent defaults, and immutable-question assertions, replacing `SearchAgentState(...)` with `_state(...)`.

- [ ] **Step 2: Add orchestration compatibility assertions**

Extend `test_agent_state_keeps_runtime_fields_structured_and_slotted`:

```python
assert state.question == "Find docs"

_[Section compacted.]_

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

Add this helper to `tests/unit/test_components.py` and replace every `SearchAgentState(question="q")` call:

```python
def _state(question: str = "q") -> AgentState:
    return AgentState(
        request_id="req-1",
        user_request=UserRequest(user_id="u1", channel="test", message=question),
        question=question,
    )
```

Update Planner assertions to pass state:

```python
state = _state()
state.record_search_round(["already searched"], [])
decision = Planner().decide("<search>already searched</search>", state)
assert isinstance(decision, SearchAction)
assert decision.is_duplicate is True
```

- [ ] **Step 2: Add a round-level Search Tool test**

```python
@pytest.mark.asyncio
async def test_search_tool_records_parallel_queries_as_one_round() -> None:
    calls: list[list[str]] = []

    async def retrieve(queries: list[str]) -> list[list[SearchResult]]:
        calls.append(queries)
        return [[_result("a")], [_result("b")]]

    state = _state()

_[Section compacted.]_

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

```python
def test_planner_parses_complete_mixed_turn() -> None:
    text = (
        "<think>plan</think>"
        "<subquestions>T1: first</subquestions>"
        '<searches retriever="web" rerank="true"><query>[T1] q</query></searches>'
        "<fetch>https://example.com</fetch>"
        "<answer>candidate</answer>"
    )
    planner = Planner()
    assert planner.parse_actions(
        text, ["think", "subquestions", "search", "searches", "fetch", "answer"]
    ) == [
        ("think", "plan"),
        ("subquestions", "T1: first"),
        ("searches", "<query>[T1] q</query>"),
        ("fetch", "https://example.com"),
        ("answer", "candidate"),
    ]
    assert planner.round_retriever(text, ["search", "searches"]) is Retriever.WEB
    assert planner.round_rerank(text, ["search", "searches"]) is True
```

- [ ] **Step 2: Run the parity test and confirm failure**

Run: `pytest tests/unit/test_components.py::test_planner_parses_complete_mixed_turn -v`

Expected: FAIL because the three Planner methods do not exist.

- [ ] **Step 3: Move the generic action parsing into Planner**

_[Section compacted.]_

### Task 4: Thread one `AgentState` through the live loop

**Files:**
- Modify: `src/agents/search.py`
- Modify: `tests/unit/test_agent_loop.py`
- Modify: `tests/unit/test_components.py`

**Interfaces:**
- Consumes: Task 1 `AgentState`; Task 2 component APIs; Task 3 Planner parsing
- Produces: unchanged `SearchAgentLoop.run(...) -> AgentLoopOutput`
- Produces: `Planner.partition_search_requests(query_specs, state, effective_limit) -> tuple[allowed, repeated, overflow]`

- [ ] **Step 1: Add a state-wiring regression trajectory**

Use the existing fake tokenizer/server fixtures. Drive two model turns: first a parallel search with rerank requested, then a cited answer. Inject fake retrieval, reranking, and evaluation dependencies. Assert:

```python
assert output.metrics["search_rounds"] == 1.0
assert output.metrics["rounds_used"] == 1.0
assert output.metrics["evidence_score_final"] > 0.0
assert output.final_answer == "Grounded [R1Q1D1]."
assert output.context.num_rounds == 1
```

Also capture the state passed to Search Tool, Reranker Tool, Evidence Judge, and Answer Generator and assert all captured object IDs are equal.

- [ ] **Step 2: Run the wiring test and confirm failure**

Run: `pytest tests/unit/test_agent_loop.py::test_search_loop_threads_one_agent_state_through_components -v`

Expected: FAIL because the live loop does not construct or pass `AgentState`.

- [ ] **Step 3: Construct state once at the start of `run`**

Add a private pure helper for question extraction, returning the last non-empty user message or an empty string:

```python
@staticmethod
def _question_from_messages(messages: list[dict[str, Any]]) -> str:

_[Section compacted.]_

### Task 5: Remove duplicate state and verify compatibility

**Files:**
- Modify: `src/agents/state.py`
- Modify: `src/agents/__init__.py` only if it exports the removed name
- Modify: `tests/unit/test_agent_state.py`
- Modify: `tests/unit/test_state_models.py`
- Modify: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Produces: one exported concrete `src.agents.state.AgentState`
- Preserves: graph-local `src.agents.graph_base.AgentState` TypedDict; it is unrelated and remains unchanged

- [ ] **Step 1: Add a repository-level no-duplicate assertion**

Add to `tests/unit/test_agent_state.py`:

```python
def test_search_agent_state_is_not_exported() -> None:
    import src.agents.state as state_module

    assert not hasattr(state_module, "SearchAgentState")
    assert "SearchAgentState" not in state_module.__all__
```

- [ ] **Step 2: Run the complete focused suite**

Run:

```bash
pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py tests/unit/test_components.py tests/unit/test_agent_loop.py tests/unit/test_reward.py -v
```

Expected: PASS.

- [ ] **Step 3: Scan for stale references and duplicate canonical locals**

Run:

```bash
rg -n "SearchAgentState" src tests examples
rg -n "rounds_used =|executed_queries:|evidence_score_final.*=" src/agents/search.py
```

Expected: first command has no output. The second may show metrics assignment only; it must not show independent canonical state initialization or mutation.

- [ ] **Step 4: Run static checks**

Run:

```bash
ruff check src/agents tests/unit/test_agent_state.py tests/unit/test_state_models.py tests/unit/test_components.py tests/unit/test_agent_loop.py

_[Section compacted.]_

### Future Extension (Not an Implementation Task)

The final workflow leaves `LoopController` as the future policy seam for human-in-the-loop control. A separate design may later introduce a `PAUSE_FOR_HUMAN` decision for high-risk, side-effecting, or unusually expensive actions. It must define persistence, timeout, idempotent resume, and UI/API contracts before implementation; none of those behaviors belong in this plan.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
