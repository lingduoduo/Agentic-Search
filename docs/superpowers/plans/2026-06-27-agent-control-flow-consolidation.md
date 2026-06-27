# Agent Control-Flow Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the existing automated search workflow by making the exported `AgentState`, named components, and Loop Controller the authoritative execution path.

**Architecture:** Merge the six existing search fields and transition methods into `AgentState`, then delete `SearchAgentState`. Route parsing, retrieval, reranking, evidence evaluation, answer handling, and continue/stop policy through their existing named components while retaining `AgentContext` solely for evidence formatting and output compatibility.

**Tech Stack:** Python 3.11+, slotted dataclasses, asyncio, pytest, Ruff.

## Global Constraints

- Do not introduce another state class, compatibility wrapper, or alias.
- Preserve `SearchAgentLoop.run(...)`, `AgentLoopOutput`, the XML protocol, mixed-action behavior, metrics, retries, caching, result ordering, answer gating, and forced answers.
- `search_rounds` counts completed search rounds; parallel queries in one round increment it once.
- Keep existing `AgentState` orchestration fields and construction compatible.
- Add no dependency, endpoint, configuration setting, or model action.
- Keep the workflow fully automated; do not add human approval, pause/resume state, or user-response handling.
- Follow TDD for every behavior change and commit after each task passes its focused tests.

---

## File Map

- `src/agents/state.py` — the one concrete `AgentState`; remove `SearchAgentState`.
- `src/agents/components/planner.py` — parse a complete existing XML turn and consult state query history.
- `src/agents/components/search_tool.py` — execute/cache a complete search round and record it once.
- `src/agents/components/reranker_tool.py` — accept `AgentState` and own rerank mutation.
- `src/agents/components/evidence_judge.py` — accept `AgentState` and own evidence-score mutation.
- `src/agents/components/answer_generator.py` — accept `AgentState` and own citation mutation.
- `src/agents/search.py` — construct one state, wire all components, and derive loop metrics from it.
- `tests/unit/test_agent_state.py` — canonical search-field invariants on `AgentState`.
- `tests/unit/test_state_models.py` — orchestration compatibility and serialization.
- `tests/unit/test_components.py` — component interfaces and mutations.
- `tests/unit/test_agent_loop.py` — behavior-regression trajectories for consolidated wiring.

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
assert state.previous_queries == []
assert state.retrieved_docs == []
assert state.evidence_score == 0.0
assert state.search_rounds == 0
assert state.citations == []
assert state.to_dict()["question"] == "Find docs"
```

- [ ] **Step 3: Run the tests and confirm the expected failure**

Run: `pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py -v`

Expected: FAIL because `AgentState` does not accept `question` and has no `record_search_round`.

- [ ] **Step 4: Merge the fields and methods into `AgentState`**

In `src/agents/state.py`, keep `request_id` and `user_request` required, add the six fields after them, and use `__post_init__` only to default the question from the existing request:

```python
@dataclass(slots=True)
class AgentState:
    request_id: str
    user_request: UserRequest
    question: str = ""
    previous_queries: list[str] = field(default_factory=list)
    retrieved_docs: list[SearchResult] = field(default_factory=list)
    evidence_score: float = 0.0
    search_rounds: int = 0
    citations: list[Citation] = field(default_factory=list)
    # existing orchestration fields follow unchanged

    def __post_init__(self) -> None:
        if not self.question:
            self.question = self.user_request.message

    def record_search_round(
        self, queries: list[str], docs: list[SearchResult]
    ) -> None:
        for query in queries:
            if query not in self.previous_queries:
                self.previous_queries.append(query)
        self.retrieved_docs.extend(docs)
        self.search_rounds += 1

    def record_rerank(self, reordered_docs: list[SearchResult]) -> None:
        self.retrieved_docs = list(reordered_docs)

    def set_evidence(self, score: float) -> None:
        self.evidence_score = max(0.0, min(1.0, score))

    def set_citations(self, citations: list[Citation]) -> None:
        self.citations = list(citations)
```

Move `Retriever` and `Citation` above `AgentState` so its annotations resolve. Leave `SearchAgentState` temporarily intact so the repository remains green until its component callers migrate in Task 2.

- [ ] **Step 5: Run focused state tests**

Run: `pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the state consolidation**

```bash
git add src/agents/state.py tests/unit/test_agent_state.py tests/unit/test_state_models.py
git commit -m "refactor: consolidate agent state models"
```

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
    rows = await SearchTool(retrieve).run_round(state, ["one", "two"])

    assert calls == [["one", "two"]]
    assert len(rows) == 2
    assert state.previous_queries == ["one", "two"]
    assert state.search_rounds == 1
```

- [ ] **Step 3: Run component tests and confirm failure**

Run: `pytest tests/unit/test_components.py -v`

Expected: FAIL on removed `SearchAgentState`, Planner's old signature, and missing `run_round`.

- [ ] **Step 4: Replace component annotations and Planner input**

In all five modules import `AgentState`, not `SearchAgentState`. Change Planner to read query history from state:

```python
def decide(self, text: str, state: AgentState) -> PlannerDecision:
    seen = {_normalize_query(query) for query in state.previous_queries}
    # retain the existing parsing and fallback logic unchanged
```

- [ ] **Step 5: Add round execution to Search Tool**

Change its injected callable to a batch interface and record only after retrieval returns:

```python
RetrieveFn = Callable[[list[str]], Awaitable[list[list[SearchResult]]]]

async def run_round(
    self,
    state: AgentState,
    queries: list[str],
    retriever: Retriever = Retriever.VECTOR_DB,
) -> list[list[SearchResult]]:
    rows = await self._retrieve(retriever, queries)
    state.record_search_round(queries, [doc for row in rows for doc in row])
    return rows
```

Keep cache keys as `(retriever, normalized_query)`. Retrieve only uncached queries, reconstruct rows in input order, and preserve the existing web-to-vector fallback. Match the live loop's current terminal-failure behavior: log the error, substitute one empty row per failed query, and record the attempted round. Do not increment state when `queries` is empty.

- [ ] **Step 6: Run component tests**

Run: `pytest tests/unit/test_components.py -v`

Expected: PASS.

- [ ] **Step 7: Prove the duplicate class name is gone from components and tests**

Run: `rg -n "SearchAgentState" src/agents/components tests/unit/test_agent_state.py tests/unit/test_components.py`

Expected: no output and exit status 1.

- [ ] **Step 8: Delete the migrated duplicate class**

Delete `SearchAgentState` from `src/agents/state.py` and remove it from `__all__`. Run:

```bash
rg -n "SearchAgentState" src/agents/state.py src/agents/components tests/unit/test_agent_state.py tests/unit/test_components.py
pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py tests/unit/test_components.py -v
```

Expected: the scan has no output; all tests pass.

- [ ] **Step 9: Commit component migration**

```bash
git add src/agents/state.py src/agents/components tests/unit/test_agent_state.py tests/unit/test_components.py
git commit -m "refactor: thread agent state through search components"
```

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

Implement the three parsing methods with escaped configured tag names and the same case sensitivity as the current loop. Compile regexes inside the methods only from the small configured tag list; do not introduce configuration state in Planner. Preserve absent/unknown retriever as `Retriever.VECTOR_DB` and recognize rerank only when its value is `true`.

- [ ] **Step 4: Replace loop parser calls**

Construct `self._planner = Planner()` in `SearchAgentLoop.__init__`. Replace `_parse_actions`, `_parse_round_retriever`, and `_parse_round_rerank` call sites with the new Planner methods, including `_force_final_answer`. Delete the three duplicate loop methods and their constructor regex fields.

- [ ] **Step 5: Run parser and loop-focused tests**

Run: `pytest tests/unit/test_components.py tests/unit/test_agent_loop.py -v`

Expected: PASS.

- [ ] **Step 6: Commit parser ownership**

```bash
git add src/agents/components/planner.py src/agents/search.py tests/unit/test_components.py tests/unit/test_agent_loop.py
git commit -m "refactor: centralize search action parsing"
```

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
    return next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(messages)
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )
```

At the beginning of `run`, create:

```python
question = self._question_from_messages(messages)
state = AgentState(
    request_id=request_id,
    user_request=UserRequest(user_id="", channel="search_agent", message=question),
    question=question,
)
```

- [ ] **Step 4: Replace canonical loop locals with state reads and writes**

Perform these mechanical replacements without changing surrounding branches:

- `rounds_used` → `state.search_rounds`
- `executed_queries` membership → normalized membership derived from `state.previous_queries`
- `metrics["evidence_score_final"]` as decision input → `state.evidence_score`
- accumulated retrieved documents → `state.retrieved_docs`

Keep `active_tasks`, `task_statuses`, rejection counters, format-error counters, caches, pages, messages, and `latest_evaluation` local because they are transient control or compatibility data, not one of the six canonical fields.

Move `_partition_search_requests` into Planner as `partition_search_requests`. Add a component test that records `"seen"` on state, partitions `[(None, "seen"), (None, "new")]`, and asserts only `"new"` is allowed. Use normalized keys from `state.previous_queries` and `state.search_rounds >= effective_limit` for overflow. Replace the loop call and delete its duplicate method.

- [ ] **Step 5: Route the search round through components**

Instantiate the five components once per loop instance, except Search Tool's per-run cache must be reset by constructing it per `run` or by an explicit `reset()` at run start. The search branch must call in this order:

```python
rows = await search_tool.run_round(state, search_tool_call.queries, retriever)
# preserve current source deduplication and SearchContext construction
if rerank and self._reranker is not None:
    reranker_tool.run(state, query="\n".join(search_tool_call.queries))
verdict = evidence_judge.update_state(state, search_contexts)
```

Ensure `AgentContext.record_round(search_contexts)` still occurs once and the observation builder sees the same row ordering and citation labels.

- [ ] **Step 6: Derive controller snapshots and metrics from state**

Use `state.search_rounds` for budgets and `state.evidence_score` for plateau snapshots. Update observability keys only after state mutation:

```python
metrics["search_rounds"] = float(state.search_rounds)
metrics["rounds_used"] = float(state.search_rounds)
metrics["evidence_score_final"] = state.evidence_score
```

- [ ] **Step 7: Route accepted and forced answers through Answer Generator**

Before returning, when `final_answer is not None`, call:

```python
answer_result = self._answer_generator.update_state(state, final_answer, agent_ctx)
final_answer = answer_result.answer
```

Do not alter the public `AgentLoopOutput` fields. Continue returning `agent_ctx` as `context`.

- [ ] **Step 8: Run loop and component regressions**

Run: `pytest tests/unit/test_agent_loop.py tests/unit/test_components.py tests/unit/test_reward.py -v`

Expected: PASS with identical fixed-trajectory metrics.

- [ ] **Step 9: Commit live-loop wiring**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py tests/unit/test_components.py
git commit -m "refactor: consolidate search loop control flow"
```

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
ruff format --check src/agents tests/unit/test_agent_state.py tests/unit/test_state_models.py tests/unit/test_components.py tests/unit/test_agent_loop.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Run the full default test suite**

Run: `pytest`

Expected: PASS. Integration tests that require external services remain outside the default suite as configured by the repository.

- [ ] **Step 6: Commit final cleanup**

```bash
git add src/agents tests/unit/test_agent_state.py tests/unit/test_state_models.py tests/unit/test_components.py tests/unit/test_agent_loop.py
git commit -m "test: verify unified agent control flow"
```

## Final Review Checklist

- [ ] Exactly one concrete runtime `AgentState` remains in `src/agents/state.py`.
- [ ] The unrelated graph TypedDict remains untouched.
- [ ] All five components and Loop Controller observe the same state instance.
- [ ] Parallel searches increment `search_rounds` once.
- [ ] Metrics are derived from state and fixed trajectories are unchanged.
- [ ] `AgentContext` output and citation labels remain compatible.
- [ ] Focused tests, full tests, Ruff, formatting, and `git diff --check` pass.

## Future Extension (Not an Implementation Task)

The final workflow leaves `LoopController` as the future policy seam for human-in-the-loop control. A separate design may later introduce a `PAUSE_FOR_HUMAN` decision for high-risk, side-effecting, or unusually expensive actions. It must define persistence, timeout, idempotent resume, and UI/API contracts before implementation; none of those behaviors belong in this plan.
