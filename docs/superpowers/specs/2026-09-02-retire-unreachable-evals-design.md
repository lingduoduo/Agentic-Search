# Retire the unreachable evals cluster

Date: 2026-09-02

## Problem

`src/internal/servers/evals/` holds two unrelated things under one package name.

**Live:** `api.py` (207 lines) builds the evals router mounted by
`create_web_app`. It runs `run_expanded_search` synchronously and serves
`/evals/eval_run`, `/evals/eval_run_ack` and `/api/admin/evals/summary`. Three
test modules cover it.

**Unreachable:** everything else — `eval.py`, `eval_cli.py`, `models.py`,
`provider.py`, `providers/` and `one_off/` — 922 lines of Braintrust-era
heritage that cannot run:

- `eval_cli` imports `eval`, which imports `provider`, which imports
  `providers.braintrust`, which does `from braintrust import Eval`. `braintrust`
  is in neither `requirements.txt` nor `pyproject.toml`, so the chain raises
  `ModuleNotFoundError` at import. `python3 -m src.internal.servers.evals.eval_cli --help`
  fails before parsing an argument.
- If it did import, `run_eval` dispatches to `_get_answer_with_tools`, which
  raises `NotImplementedError("requires src.internal.db — wire before use")`.
  So does `_get_multi_turn_answer_with_tools`. Those are the only two tasks the
  runner has.
- `one_off/create_braintrust_dataset.py` calls `sys.exit(1)` at import time when
  `braintrust` is missing, and reads a CSV ("DR Master Question & Metric Sheet")
  that is not in this repository.

`api.py` imports nothing from the cluster. The two halves share only the package
name — and a name: each defines a different `EvalConfigurationOptions`, one a
query plus `num_hits`, the other a chat-eval configuration.

Nothing outside the package references the cluster. Every apparent hit is a
substring match (`retrieval_client` for `eval_cli`, `run_search` for `run_eval`,
`get_provider` in `src/internal/tools/api.py`). The only true mentions are in
`docs/superpowers/archive/` — history, which stays as written.

The repository already has working eval harnesses: `src/model/post_training/eval/`,
`examples/run_bamboogle_eval.py`, and the Dev Console's eval-results panel.

## Design

Delete `eval.py`, `eval_cli.py`, `models.py`, `provider.py`, `providers/` and
`one_off/`. Keep `__init__.py` and `api.py`.

### Why delete rather than wire

Wiring it means implementing an eval against `src/internal/chat/`, the
Onyx-heritage stack that is not the Assist path, and taking a dependency on a
commercial SDK the project does not declare — to duplicate harnesses that already
work. Removing 922 lines that cannot execute is the smaller and more honest
change.

### An import guard, so this cannot recur

The failure that let this rot is that an un-importable module can sit in the tree
indefinitely: nothing imports it, so nothing notices. Measured across
`src/internal/servers/`, **exactly 5 of 154 modules fail to import, and all 5 are
this cluster.**

Add `tests/unit/servers/test_server_modules_import.py`: walk every module under
`src/internal/servers/` and assert it imports. It costs ~1.5s and passes with
`torch` blocked (no server module needs torch), so it is safe in the torch-free
CI job.

The guard catches `SystemExit` as well as `Exception` — `create_braintrust_dataset`
exits rather than raising, and a bare `except Exception` would have let it through.

## Testing

- The new import guard passes over the whole server tree.
- The guard fails if an un-importable module is reintroduced (verified by adding
  one).
- The three existing evals tests still pass — they only ever touched `api.py`.
- The full suite is unchanged otherwise.

## Out of scope

`api.py` itself. Its endpoints work, are mounted and are tested. The duplicated
`EvalConfigurationOptions` name resolves on its own once the other definition is
gone.
