# Remove dead examples/evaluate_bamboogle.py

**Date:** 2026-07-19
**Status:** Approved

## Problem

`examples/evaluate_bamboogle.py` is a non-functional scaffold: its `_build_agent`
helper unconditionally raises `NotImplementedError` (line 86), so `main()` can
never run. It also imports pre-refactor paths (`src.agents.graph_base`,
`src.agents.base`) that no longer match the current agents package layout. The
working Bamboogle eval CLI is `examples/run_bamboogle_eval.py`.

Verified unreferenced: the only references to the *name* `evaluate_bamboogle` in
tests and code point at the **function** in `src/training/eval/bamboogle.py`
(imported at `tests/unit/test_bamboogle_eval.py:14`), not this example script.
The script is referenced only inside its own docstring.

## Design

Delete `examples/evaluate_bamboogle.py`. No code imports it; the reusable eval
logic lives in `src/training/eval/bamboogle.py` and the working CLI is
`examples/run_bamboogle_eval.py`, both untouched.

## Scope / non-goals

- Do not touch `src/training/eval/bamboogle.py` or `run_bamboogle_eval.py`.
- No behavior change; this only removes an unreachable dead file.

## Verification

- `grep` shows no import of `examples.evaluate_bamboogle` anywhere.
- `pytest tests/unit/test_bamboogle_eval.py` stays green (it uses the src function).
