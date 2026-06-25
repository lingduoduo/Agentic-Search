# Agentic Search Control Flow Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit lifecycle outcome metrics and repeated no-action stopping to `SearchAgentLoop`, borrowing the useful control-flow ideas from the sampled Basic agent without adding dependencies or changing public APIs.

**Architecture:** Keep `SearchAgentLoop.run(...)` as the public orchestration point and add numeric exit counters to its existing metrics dictionary. Implement the smallest internal control-flow changes first, then remove the appended Basic agent sample from `examples/run_agentic_search.py` after its ideas are captured in the spec and this plan.

**Tech Stack:** Python 3.11+, existing async `SearchAgentLoop`, pytest unit tests, no new dependencies.

## Global Constraints

- The sample code is reference material and must not remain appended to `examples/run_agentic_search.py` as production CLI code.
- Public agent APIs stay stable; `SearchAgentLoop.run(...)` still returns `AgentLoopOutput`.
- The XML action tags stay unchanged.
- Evidence sufficiency and answer rejection rules stay unchanged.
- This pass optimizes control flow observability and maintainability, not model policy, reward weights, or retrieval quality.
- Represent exit status with numeric metrics only, because training/reward code treats metrics as `dict[str, float]`.

---

## File Structure

- Modify `src/agents/search.py`: add config, metrics, no-action control flow, and helper methods for terminal exit counters.
- Modify `tests/unit/test_agent_loop.py`: add focused tests for answered exit, max-turn exit, search-limit exit, and repeated no-action format-error exit.
- Modify `examples/run_agentic_search.py`: remove the appended Basic agent sample and leave the original CLI entrypoint intact.
- Keep `src/training/reward.py` unchanged; new metrics are observational only.

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
        server_manager=DummyServerManager([tokenizer.encode("<think>plan</think>")]),
        search_config=SearchAgentLoopConfig(max_turns=1),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research"}], {"temperature": 0.0})
    )

    assert output.final_answer is None
    assert output.num_turns == 1
    assert output.metrics["exit_answered"] == 0.0
    assert output.metrics["exit_max_turns"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "answered_exit_metric or max_turns_exit_metric" -v
```

Expected: both tests fail with `KeyError` for missing exit metrics.

- [ ] **Step 3: Add exit metric defaults and helper**

In `src/agents/search.py`, extend `_initial_metrics()`:

```python
"exit_answered": 0.0,
"exit_max_turns": 0.0,
"exit_search_limit": 0.0,
"exit_format_error_limit": 0.0,
"exit_no_action": 0.0,
"exit_exception": 0.0,
```

Add this helper method near `_initial_metrics()`:

```python
@staticmethod
def _mark_exit(metrics: dict[str, float], name: str) -> None:
    metrics[f"exit_{name}"] = 1.0
```

- [ ] **Step 4: Set answered and max-turn exits**

In `SearchAgentLoop.run`, introduce local state before the turn loop:

```python
exit_status: str | None = None
```

When a direct answer is accepted, set it before `break`:

```python
exit_status = "answered"
```

When a gated answer is accepted, set the same status before `break`.

After the loop and before derived metrics, add:

```python
if exit_status is None:
    exit_status = "max_turns"
self._mark_exit(metrics, exit_status)
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "answered_exit_metric or max_turns_exit_metric" -v
```

Expected: both tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py docs/superpowers/plans/2026-06-25-agentic-search-control-flow-optimization.md
git commit -m "feat: record search loop exit metrics"
```

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
                tokenizer.encode("<search>alpha</search>"),
                tokenizer.encode("<search>beta</search>"),
            ]
        ),
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            max_search_limit=1,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    loop._search_client = FakeSearchClient({("alpha",): [[]]})

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research"}], {"temperature": 0.0})
    )

    assert output.metrics["search_limit_hits"] == 1.0
    assert output.metrics["search_budget_exhausted_without_answer"] == 1.0
    assert output.metrics["exit_search_limit"] == 1.0
    assert output.metrics["exit_max_turns"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "repeated_no_action_turns or search_limit_exit_metric" -v
```

Expected: the no-action test fails because `max_consecutive_format_errors` and `format_error_turns` are missing; the search-limit test fails because `exit_search_limit` is not set.

- [ ] **Step 3: Add config and metric defaults**

In `SearchAgentLoopConfig`, add after `max_answer_rejections`:

```python
# Stop after this many consecutive no-action/format-error turns. 0 disables.
max_consecutive_format_errors: int = 3
```

In `_initial_metrics()`, add:

```python
"format_error_turns": 0.0,
```

- [ ] **Step 4: Implement no-action counter and exit**

In `SearchAgentLoop.run`, add local state before the loop:

```python
consecutive_format_errors = 0
```

In the `if not actions:` block, increment first:

```python
consecutive_format_errors += 1
metrics["format_error_turns"] += 1.0
```

Then, before computing `needs_more`, stop if the configured limit is reached:

```python
if (
    cfg.max_consecutive_format_errors > 0
    and consecutive_format_errors >= cfg.max_consecutive_format_errors
):
    exit_status = "format_error_limit"
    break
```

After any turn with recognized actions, reset:

```python
consecutive_format_errors = 0
```

- [ ] **Step 5: Set search-limit and no-action exits**

When the existing no-action path exits because no more feedback is needed, set:

```python
exit_status = "no_action"
```

When final metrics detect `search_budget_exhausted_without_answer`, set `exit_status = "search_limit"` before `_mark_exit(...)` if no earlier exit status was set:

```python
if (
    exit_status == "max_turns"
    and metrics["search_budget_exhausted_without_answer"] > 0.0
):
    exit_status = "search_limit"
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
pytest tests/unit/test_agent_loop.py -k "repeated_no_action_turns or search_limit_exit_metric" -v
```

Expected: both tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py docs/superpowers/plans/2026-06-25-agentic-search-control-flow-optimization.md
git commit -m "feat: bound no-action search loop turns"
```

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
git add examples/run_agentic_search.py tests/unit/test_run_agentic_search.py docs/superpowers/plans/2026-06-25-agentic-search-control-flow-optimization.md
git commit -m "chore: remove sampled agent code from cli"
```

## Final Verification

- [ ] Run `git diff --check`.
- [ ] Run `pytest tests/unit/test_agent_loop.py -k "exit or format_error or search_limit" -v`.
- [ ] Run `pytest tests/unit/test_agent_loop.py -v`.
- [ ] Run `pytest tests/unit/test_run_agentic_search.py -v`.
- [ ] Run `git status --short` and confirm only intended files are changed or the branch is clean.
