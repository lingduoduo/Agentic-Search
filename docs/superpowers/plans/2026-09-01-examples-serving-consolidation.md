# Plan: examples serving consolidation and concurrency fixes

Spec: `docs/superpowers/specs/2026-09-01-examples-serving-consolidation-design.md`

## Task 1 — `LocalServerManager.aclose`

1. Add a failing test in `tests/unit/test_model_serving.py`: `LocalServerManager`
   has a callable `aclose`, and awaiting it on an instance built without loading
   a model returns `None`.
   → verify: test fails with `AttributeError` / missing attribute.
2. Add `async def aclose(self) -> None` to `LocalServerManager`.
   → verify: new test passes.
3. Drop the `hasattr(server_manager, "aclose")` guard in
   `examples/run_agentic_search.py` and call `aclose()` directly.
   → verify: `pytest tests/unit/test_run_agentic_search.py` green.
4. Remove the unguarded-call hazard: `run_bamboogle_eval` and
   `run_bamboogle_synthetic_grpo` now work in `--local` mode unchanged.
   → verify: a test drives `_SyncAgent.invoke` with a stub local manager and
   gets the loop result back rather than an `AttributeError`.

## Task 2 — loop-bound aiohttp session

1. Add a failing test: four threads each `asyncio.run` a coroutine that calls
   `_get_session()` on one shared `OpenAIServerManager` and issues a request to
   a local stub HTTP server; assert all four succeed.
   → verify: test fails with `Timeout context manager ...` / different-loop error.
2. Track `self._session_loop` and rebuild the session when
   `asyncio.get_running_loop()` differs.
   → verify: new test passes; `test_model_serving.py` still green.

## Task 3 — one server-manager builder for the bamboogle scripts

1. Add `build_server_manager_from_args(args, tokenizer)` to
   `examples/run_bamboogle_eval.py`, delegating to
   `src.model.serving.build_server_manager` and forwarding `dtype` when present.
2. Replace `_build_server_manager` with it; keep no alias.
3. Point `run_bamboogle_synthetic_grpo._build_loop_factory` at the public name.
   → verify: test asserts no `_build_server_manager` import anywhere; both
   scripts `--help` cleanly.

## Task 4 — one search-loop builder

1. Add `build_search_loop(args, tokenizer, server_manager)` to
   `run_bamboogle_eval.py` holding the shared `SearchAgentLoop` construction.
2. Call it from `_build_agent` and from
   `run_bamboogle_synthetic_grpo._build_loop_factory`.
   → verify: the duplicate-block scan reports no shared 8-line block between the
   two scripts; `pytest tests/unit/test_bamboogle_eval.py` green.

## Task 5 — `run_retriever_aware_grpo` refresh

1. Rewrite the stale "until PR #326 merges" dependency note.
2. `asyncio.get_event_loop().run_in_executor(None, fn, *a)` → `asyncio.to_thread(fn, *a)`.
3. `_eval_side` stays sequential — see the spec: the rollouts share one policy
   on one device, so they are compute-bound, not I/O-bound.
   → verify: no change; the script still imports and `--help` runs.

## Task 6 — verification

1. `pytest` (full default suite).
2. Torch-free run: block `torch` via `sys.meta_path` and confirm no new skips.
3. `ruff check . --fix && ruff format .`
4. `python3 -m examples.run_bamboogle_eval --help` and
   `python3 -m examples.run_bamboogle_synthetic_grpo --help` (direct invocation
   catches the `src.` import trap from #558).
5. Mutation-check the two bug tests: revert each fix, confirm the test goes red,
   restore, and clear stale `.pyc` before re-running.
