# Plan: retire the unreachable evals cluster

Spec: `docs/superpowers/specs/2026-09-02-retire-unreachable-evals-design.md`

## Task 1 — the import guard, written against the current tree

1. Add `tests/unit/servers/test_server_modules_import.py`: import every module
   under `src/internal/servers/`, catching `BaseException` so a `sys.exit` at
   import time is reported rather than killing the run.
   → verify: it FAILS now, naming exactly the 5 cluster modules.

## Task 2 — delete the cluster

1. `git rm` `eval.py`, `eval_cli.py`, `models.py`, `provider.py`, `providers/`,
   `one_off/`.
   → verify: the guard passes; `pytest tests/unit/servers/evals/` still passes.

## Task 3 — verification

1. `pytest` (full default suite).
2. Torch-free run of the guard with `torch` blocked via `sys.meta_path`.
3. Confirm the web app still builds and the evals router still serves:
   `create_web_app` + a request to `/api/admin/evals/summary`.
4. `ruff check . --fix && ruff format .`
5. Mutation-check the guard: add a module that raises on import, confirm red,
   remove it. Clear stale `.pyc` before re-running.
