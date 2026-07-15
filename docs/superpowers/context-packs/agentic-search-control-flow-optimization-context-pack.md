# Generated Context Pack

# Agentic Search Control Flow Optimization

## Sources

- [Specification: 2026-06-25-agentic-search-control-flow-optimization-design.md](../specs/2026-06-25-agentic-search-control-flow-optimization-design.md)
- [Plan: 2026-06-25-agentic-search-control-flow-optimization.md](../plans/2026-06-25-agentic-search-control-flow-optimization.md)

## Specification Context

### Testing Strategy

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

### Out of Scope

- Importing or depending on `minisweagent`.
- Changing the XML action vocabulary.
- Changing reward weights.
- Changing retrieval, reranking, evidence scoring, or citation behavior.
- Replacing `SearchAgentLoop` with a full state-machine architecture.

## Implementation Plan Context

### Global Constraints

- The sample code is reference material and must not remain appended to `examples/run_agentic_search.py` as production CLI code.
- Public agent APIs stay stable; `SearchAgentLoop.run(...)` still returns `AgentLoopOutput`.
- The XML action tags stay unchanged.
- Evidence sufficiency and answer rejection rules stay unchanged.
- This pass optimizes control flow observability and maintainability, not model policy, reward weights, or retrieval quality.
- Represent exit status with numeric metrics only, because training/reward code treats metrics as `dict[str, float]`.

---

### Task 1: Terminal Exit Metrics

**Files:**
- Modify: `src/agents/search.py:180-233`, `src/agents/search.py:342-370`, `src/agents/search.py:924-1260`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: existing `SearchAgentLoop.run(messages, sampling_params, *, on_turn=None) -> AgentLoopOutput`
- Produces: numeric metrics `exit_answered`, `exit_max_turns`, `exit_search_limit`, `exit_format_error_limit`, `exit_no_action`, `exit_exception`

- [ ] **Step 1: Write failing tests for answered and max-turn exits**

Add these tests near other `SearchAgentLoop` tests in `tests/unit/test_agent_loop.py`:

```python
def test_search_agent_loop_records_answered_exit_metric():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            [tokenizer.encode("<answer>direct answer</answer>")]
        ),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            require_sufficient_evidence_before_answer=False,
        ),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.0})
    )

    assert output.final_answer == "direct answer"
    assert output.metrics["exit_answered"] == 1.0
    assert output.metrics["exit_max_turns"] == 0.0
    assert output.metrics["exit_search_limit"] == 0.0
    assert output.metrics["exit_no_action"] == 0.0
```

```python
def test_search_agent_loop_records_max_turns_exit_metric():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,

_[Section compacted.]_

### Task 2: No-Action Format Error Limit And Search-Limit Exit

**Files:**
- Modify: `src/agents/search.py:180-233`, `src/agents/search.py:342-370`, `src/agents/search.py:950-985`, `src/agents/search.py:1160-1225`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `SearchAgentLoopConfig`
- Produces: config field `max_consecutive_format_errors: int = 3`; metrics `format_error_turns`, `exit_format_error_limit`, and `exit_search_limit`

- [ ] **Step 1: Write failing tests for no-action and search-limit exits**

Add these tests to `tests/unit/test_agent_loop.py`:

```python
def test_search_agent_loop_stops_after_repeated_no_action_turns():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            [
                tokenizer.encode("no tags here"),
                tokenizer.encode("still no tags"),
            ]
        ),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_consecutive_format_errors=2,
        ),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research"}], {"temperature": 0.0})
    )

    assert output.num_turns == 2
    assert output.metrics["format_error_turns"] == 2.0
    assert output.metrics["exit_format_error_limit"] == 1.0
    assert output.metrics["exit_max_turns"] == 0.0
```

```python
def test_search_agent_loop_records_search_limit_exit_metric():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            [

_[Section compacted.]_

### Task 3: Remove Sample From CLI File

**Files:**
- Modify: `examples/run_agentic_search.py`
- Test: `tests/unit/test_run_agentic_search.py`

**Interfaces:**
- Consumes: existing CLI module ending at `if __name__ == "__main__": asyncio.run(main())`
- Produces: no imported `minisweagent` symbols in `examples/run_agentic_search.py`

- [ ] **Step 1: Write failing test for sample removal**

Add this test to `tests/unit/test_run_agentic_search.py`:

```python
def test_run_agentic_search_has_no_minisweagent_dependency():
    source = Path("examples/run_agentic_search.py").read_text(encoding="utf-8")

    assert "minisweagent" not in source
    assert "class DefaultAgent" not in source
```

If `Path` is not imported in that file, add:

```python
from pathlib import Path
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_run_agentic_search.py -k "minisweagent_dependency" -v
```

Expected: fail because the sampled Basic agent code is still appended to `examples/run_agentic_search.py`.

- [ ] **Step 3: Remove appended sample code**

Edit `examples/run_agentic_search.py` so the file ends at:

```python
if __name__ == "__main__":
    asyncio.run(main())
```

Remove everything after that block, including imports from `minisweagent`, `jinja2`, `pydantic`, `Path`, `traceback`, and the `AgentConfig` / `DefaultAgent` definitions.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
pytest tests/unit/test_run_agentic_search.py -k "minisweagent_dependency" -v
```

Expected: pass.

- [ ] **Step 5: Commit Task 3**

```bash

_[Section compacted.]_

### Final Verification

- [ ] Run `git diff --check`.
- [ ] Run `pytest tests/unit/test_agent_loop.py -k "exit or format_error or search_limit" -v`.
- [ ] Run `pytest tests/unit/test_agent_loop.py -v`.
- [ ] Run `pytest tests/unit/test_run_agentic_search.py -v`.
- [ ] Run `git status --short` and confirm only intended files are changed or the branch is clean.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
