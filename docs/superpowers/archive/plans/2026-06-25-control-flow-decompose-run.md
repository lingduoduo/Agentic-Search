# Decompose SearchAgentLoop.run() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract two cohesive, control-flow-free blocks out of the ~500-line `SearchAgentLoop.run()` into focused private methods (`_generate_turn`, `_finalize_run_metrics`), behavior-preserving.

**Architecture:** Two pure method extractions on `SearchAgentLoop` in `src/agents/search.py`. `run()` calls them; no control-flow change, no new state type, `metrics` dict byte-identical. The existing test suite is the behavior-preserving proof.

**Tech Stack:** Python 3, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-25-control-flow-decompose-run-design.md`

## Global Constraints

- **Behavior-preserving.** No existing test may change. If an assertion shifts, the extraction changed behavior — fix the extraction, not the test.
- **`metrics` dict byte-identical.** Keys, values, and computation stay the same; `reward.py` / `action_eval.py` consume these keys. The only change is *where* the code lives.
- **`AgentLoopOutput` fields unchanged.**
- **No control-flow change.** Only blocks with no `break`/`continue` are extracted. The `for turn` loop, answer-gate, observation assembly, dead-end handling, forced-answer hook, and `finally:` stay in `run()`.
- **Verbatim move + dedent only.** The extracted bodies are the existing lines, dedented into the method; do not "improve" them.

---

## File Structure

- **Modify** `src/agents/search.py` — add `_generate_turn` and `_finalize_run_metrics` to `SearchAgentLoop`; replace the two inline blocks in `run()` with calls.
- **Test** `tests/unit/test_agent_loop.py` — add one focused `_finalize_run_metrics` unit test; all existing tests pass unchanged.

---

### Task 1: Extract `_generate_turn`

**Files:**
- Modify: `src/agents/search.py` (`run()` per-turn block `:1001-1019`; add method near the other private helpers)
- Test: `tests/unit/test_agent_loop.py` (no new test — existing suite is the gate)

**Interfaces:**
- Produces: `async _generate_turn(self, working_messages, sampling_params, request_id, turn, metrics) -> tuple[list[int], list[int], str, list[tuple[str, str]]]` returning `(prompt_ids, response_ids, response_text, actions)`.

- [ ] **Step 1: Add the method**

Add to `SearchAgentLoop` (place it just before `run()`, after `_execute_search_round`):

```python
    async def _generate_turn(
        self,
        working_messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        request_id: str,
        turn: int,
        metrics: dict[str, float],
    ) -> tuple[list[int], list[int], str, list[tuple[str, str]]]:
        """Build the prompt, generate, decode, and parse actions for one turn.

        Returns (prompt_ids, response_ids, response_text, actions). Side-effect
        free on the caller's loop state; the caller applies the returned values.
        """
        with simple_timer(f"build_prompt_turn_{turn}", metrics):
            prompt_ids = await self.build_prompt_ids(working_messages)

        with simple_timer(f"generate_turn_{turn}", metrics):
            response_ids = await self.generate_response_ids(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=f"{request_id}_t{turn}",
            )

        response_text = self.decode_response_ids(response_ids)
        actions = self._parse_actions(response_text)
        logger.debug(
            "turn=%d actions=%r", turn, [(t, c[:60]) for t, c in actions]
        )
        return prompt_ids, response_ids, response_text, actions
```

- [ ] **Step 2: Replace the inline block in `run()`**

In `run()`, replace lines `:1001-1021` (the two `simple_timer` blocks through the
`working_messages.append({"role": "assistant", ...})`) with:

```python
                prompt_ids, response_ids, response_text, actions = (
                    await self._generate_turn(
                        working_messages, sampling_params, request_id, turn, metrics
                    )
                )
                final_prompt_ids = prompt_ids
                all_response_ids.extend(response_ids)
                num_turns += 1
                working_messages.append({"role": "assistant", "content": response_text})
```

Verify the surrounding lines (the `if actions:` check that follows, the `for turn` header above) are untouched and the indentation matches the loop body.

- [ ] **Step 3: Run the loop test suite (behavior-preserving gate)**

Run: `pytest tests/unit/test_agent_loop.py -q`
Expected: PASS — same count as before the change (no test modified).

- [ ] **Step 4: Commit**

```bash
git add src/agents/search.py
git commit -m "refactor: extract _generate_turn from run()"
```

---

### Task 2: Extract `_finalize_run_metrics` + unit test

**Files:**
- Modify: `src/agents/search.py` (`run()` post-loop block `:1354-1446`; add method)
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `self._loop_controller`, `self._has_sufficient_evidence`, `self._mark_exit`, `self.search_config`.
- Produces: `_finalize_run_metrics(self, metrics, *, rounds_used, task_statuses, task_search_counts, active_tasks, agent_ctx, final_answer, latest_evaluation, exit_status) -> None` (mutates `metrics`).

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_agent_loop.py
def test_finalize_run_metrics_computes_derived_keys():
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
    from src.context.search import AgentContext

    loop = SearchAgentLoop(
        tokenizer=DummyTokenizer(),  # reuse the file's existing dummy tokenizer
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_search_limit=3),
    )
    metrics = loop._initial_metrics()
    metrics["search_queries"] = 2.0
    metrics["repeated_search_queries"] = 0.0
    loop._finalize_run_metrics(
        metrics,
        rounds_used=0,
        task_statuses={},
        task_search_counts={},
        active_tasks={},
        agent_ctx=AgentContext(),
        final_answer=None,
        latest_evaluation=None,
        exit_status="answered",
    )
    # No subquestions → coverage ratio defaults to 1.0
    assert metrics["subquestion_coverage_ratio"] == 1.0
    # No answer → answer_allowed stays 0.0
    assert metrics["answer_allowed"] == 0.0
    # rounds_used surfaced as float
    assert metrics["rounds_used"] == 0.0
    # exit fixup recorded
    assert metrics["exit_answered"] == 1.0
```

(Use the actual dummy tokenizer/server-manager class names already in
`test_agent_loop.py` — read the file's top to match `DummyTokenizer`/`DummyServerManager`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_agent_loop.py::test_finalize_run_metrics_computes_derived_keys -v`
Expected: FAIL — `AttributeError: '_finalize_run_metrics'`.

- [ ] **Step 3: Add the method (verbatim move from `run()`)**

Add to `SearchAgentLoop` (after `_generate_turn`):

```python
    def _finalize_run_metrics(
        self,
        metrics: dict[str, float],
        *,
        rounds_used: int,
        task_statuses: dict[str, bool],
        task_search_counts: dict[str, int],
        active_tasks: dict[str, str],
        agent_ctx: "AgentContext",
        latest_evaluation: "SearchRoundEvaluation | None",
        final_answer: str | None,
        exit_status: str,
    ) -> None:
        """Compute the derived/reward metrics and finalize exit status, in place."""
        cfg = self.search_config
        # --- MOVE: paste search.py:1354-1446 here verbatim, dedented one level. ---
        # That block begins with `total_attempted = metrics["search_queries"] + ...`
        # and ends with `self._mark_exit(metrics, exit_status)`. It already
        # references exactly: metrics, total_attempted, task_statuses,
        # task_search_counts, rounds_used, cfg, agent_ctx, final_answer,
        # latest_evaluation, active_tasks, exit_status — all now method-local
        # (params) or self.* — so no name needs changing.
```

Move the block at `search.py:1354-1446` verbatim into the method body, dedented to method-body level. Do **not** alter any computation. The block's `cfg` references resolve to the local `cfg = self.search_config`.

- [ ] **Step 4: Replace the inline block in `run()` with the call**

In `run()`, replace lines `:1354-1446` with:

```python
        self._finalize_run_metrics(
            metrics,
            rounds_used=rounds_used,
            task_statuses=task_statuses,
            task_search_counts=task_search_counts,
            active_tasks=active_tasks,
            agent_ctx=agent_ctx,
            latest_evaluation=latest_evaluation,
            final_answer=final_answer,
            exit_status=exit_status,
        )
```

The post-loop forced-answer hook (`:1336-1352`) stays **above** this call, unchanged.

- [ ] **Step 5: Run the unit test + the loop suite**

Run: `pytest tests/unit/test_agent_loop.py -q`
Expected: PASS — the new test passes and every pre-existing test passes unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/agents/search.py tests/unit/test_agent_loop.py
git commit -m "refactor: extract _finalize_run_metrics from run() + unit test"
```

---

### Task 3: Full-suite + metrics-contract verification

**Files:**
- Test: whole unit suite

- [ ] **Step 1: Full unit suite (behavior-preserving proof)**

Run: `pytest tests/unit -q`
Expected: PASS — same pass/skip counts as `main` before this branch (no behavior change).

- [ ] **Step 2: Metrics-contract check (no key meaning changed)**

Run: `grep -oE 'metrics\.get\("[a-z_]+"' src/training/reward.py src/training/eval/action_eval.py | sort -u`
Confirm every key listed is still produced by `_finalize_run_metrics` or `_initial_metrics` (read both). Report any missing key.

- [ ] **Step 3: Lint**

Run: `ruff check . --fix && ruff format .` then re-run `pytest tests/unit -q`.

- [ ] **Step 4: Commit (only if lint changed anything)**

```bash
git add -A
git commit -m "chore: lint after run() decomposition"
```

(If nothing changed, skip.)

---

## Self-Review

**Spec coverage:** `_generate_turn` extraction (Task 1) · `_finalize_run_metrics` extraction + focused unit test (Task 2) · behavior-preserving full-suite gate + metrics-contract check (Tasks 1,2,3) · everything control-flow-entangled left in `run()` (Global Constraints). All spec sections map to a task.

**Placeholder scan:** Task 2 Step 3 uses a "MOVE: paste lines X-Y verbatim" instruction rather than reproducing ~90 lines — appropriate for a verbatim move (the source is the current `run()` body; reproducing it risks transcription drift). The signature, the call site (Step 4), and the unit test are complete and literal. Test bodies say to match the file's existing `DummyTokenizer`/`DummyServerManager` fixtures — concrete pointers, read-first.

**Type consistency:** `_generate_turn(...) -> (prompt_ids, response_ids, response_text, actions)` and `_finalize_run_metrics(..., *, rounds_used, task_statuses, task_search_counts, active_tasks, agent_ctx, latest_evaluation, final_answer, exit_status) -> None` are used identically in their call sites (Task 1 Step 2, Task 2 Step 4). Keyword-only args match between the method def and the call.
