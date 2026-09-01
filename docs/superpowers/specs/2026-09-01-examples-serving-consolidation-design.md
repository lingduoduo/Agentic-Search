# Examples: serving consolidation and concurrency fixes

Date: 2026-09-01

## Problem

An audit of `examples/*.py` against current `src/` found that every script still
imports cleanly — nothing is stale at the module level — but two documented
invocations are broken, and three scripts hand-roll construction that
`src.model.serving.build_server_manager` (#337) has owned since June.

### Bug 1 — `--local` raises on every question

`OpenAIServerManager` defines `aclose()`; `LocalServerManager` does not.
`examples/run_agentic_search.py` guards the call with `hasattr`, but
`examples/run_bamboogle_eval.py` and `examples/run_bamboogle_synthetic_grpo.py`
`await server_manager.aclose()` unconditionally. The result is an
`AttributeError` raised from a `finally` block, which also discards the
already-computed result — so `run_bamboogle_eval --local`, the "quick start,
no server needed" path in that script's own docstring, cannot complete a single
question.

The `hasattr` guard is the wrong fix: it makes "can this manager be closed?"
a per-call-site question when the answer should be uniform across the
`ServerManager` duck type.

### Bug 2 — `--concurrency > 1` fails for the server-backed path

`OpenAIServerManager` caches one `aiohttp.ClientSession` on the instance.
`evaluate_bamboogle` runs `agent.invoke` inside a `ThreadPoolExecutor`, and
`_SyncAgent.invoke` calls `asyncio.run` — so each worker thread has its own
event loop, while all of them share one manager instance and therefore one
session. An `aiohttp` session is bound to the loop that created it.

Reproduced with four threads sharing one lazily-created session: one call
succeeds and three fail with `RuntimeError: Timeout context manager should be
used inside a task` or `got Future ... attached to a different loop`.

`evaluate_bamboogle`'s own docstring recommends `concurrency` of 4–8.

### Duplication that the current factories already replaced

- `run_bamboogle_eval._build_server_manager` imports `LocalServerManager` and
  `OpenAIServerManager` **from `examples.run_agentic_search`** (which merely
  re-exports them for back-compat) and constructs them by hand, bypassing
  `build_server_manager`.
- `run_bamboogle_synthetic_grpo._build_loop_factory` imports the *private*
  `_build_server_manager` from `run_bamboogle_eval` — three hops through
  example internals to reach `src.model.serving`.
- The `SearchAgentLoop(... evaluation_config=SearchEvaluationConfig(1, 2, 10))`
  block is byte-identical in both bamboogle scripts.
- `run_retriever_aware_grpo` still carries a "until PR #326 merges, this script
  will not import" note (#325–#327 merged in June) and uses
  `asyncio.get_event_loop().run_in_executor(None, ...)` where the repo's idiom
  since #560/#561 is `asyncio.to_thread`. Its `_eval_side` awaits one
  independent greedy rollout at a time.

## Design

### `LocalServerManager.aclose`

Add `async def aclose(self) -> None: return None` to `LocalServerManager`. It
holds an in-process model with no network session, so there is nothing to
release; the method exists so every manager the factory can return closes the
same way. Remove the `hasattr` guard in `run_agentic_search` and call it
directly.

`aclose` stays off the `ServerManager` Protocol. The Protocol documents the
minimum an agent loop calls, managers are structurally typed, and every
existing test double would have to grow the method — the same reasoning that
keeps `generate_stream` off it.

### One session per event loop in `OpenAIServerManager`

Replace the single `_session` attribute with a `dict` keyed by the running
loop. `_get_session` opens this loop's session on first use; `aclose` pops and
closes only this loop's session.

Rebinding a single cached session when the loop changes is not enough. The
worker that finishes first calls `aclose`, and by then `_session` holds
whichever session was created last — so a worker would close a session another
worker is mid-request on. Keying by loop makes open and close symmetric within
each worker.

An entry is removed by `aclose`. A loop that ends without closing leaves one
behind, which is why the example callers close the manager on every path.

### Shared construction in the bamboogle scripts

`run_bamboogle_eval` grows two public helpers:

- `build_server_manager_from_args(args, tokenizer)` — one `build_server_manager`
  call, passing `torch_dtype=args.dtype` when the flag exists on the namespace.
- `build_search_loop(args, tokenizer, server_manager)` — the single
  `SearchAgentLoop` + `SearchAgentLoopConfig` + `SearchEvaluationConfig`
  construction.

`run_bamboogle_synthetic_grpo` calls both instead of reaching for a private
name. No new module: the dependency direction (synthetic → eval) already
exists, and a `_bamboogle_common.py` for two call sites would be an abstraction
for its own sake.

`run_agentic_search` keeps its own construction — it passes four extra
`SearchAgentLoopConfig` knobs and resolves the class through the agent registry,
so folding it in would mean adding parameters that only one caller uses.

### `run_retriever_aware_grpo`

Replace the stale dependency note with a statement of what the script needs at
run time. Swap `asyncio.get_event_loop().run_in_executor(None, fn, *args)` for
`asyncio.to_thread(fn, *args)`. Gather `_eval_side`'s per-question rollouts —
they share no state, and results are collected positionally, so ordering is
preserved.

## Testing

New tests pin both bugs before the fix:

- `LocalServerManager` exposes an awaitable `aclose` (fails today: no attribute).
- Four threads, each running `asyncio.run` against one shared
  `OpenAIServerManager`, all complete (fails today: 3 of 4 raise).
- Closing from one loop leaves another loop's session open, driven through the
  interleaving the thread pool produces.
- `build_server_manager_from_args` returns an `OpenAIServerManager` for
  `--server_url` and a `LocalServerManager` for `--local`.
- `run_bamboogle_synthetic_grpo` does not import any private name from
  `run_bamboogle_eval`.

Existing `tests/unit/test_model_serving.py`, `test_bamboogle_eval.py` and
`test_run_agentic_search.py` cover the refactors.

## Out of scope

`run_search_pipeline.py`'s self-contained pipeline re-implementation, teaching
the `prepare_*` scripts to accept registry corpus names via `resolve_corpus_docs`
(#558), and routing the bamboogle scripts through the agent registry.
