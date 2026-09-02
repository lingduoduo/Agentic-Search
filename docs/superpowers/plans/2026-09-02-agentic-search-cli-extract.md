# Plan: extract the parser and routing out of the agentic-search CLI

Spec: `docs/superpowers/specs/2026-09-02-agentic-search-cli-extract-design.md`

## Task 1 — confirm the seams

1. Walk the AST for the six definitions to move and list every module-level name
   each references.
   → verify: nothing beyond stdlib imports and their own classes.

## Task 2 — move

1. Create `examples/agentic_search/{__init__,parser,routing}.py`.
2. Move `_build_parser` into `parser.py`; the five routing definitions into
   `routing.py`.
3. Re-export the routing names from `run_agentic_search.py`.
   → verify: `pytest tests/unit/test_run_agentic_search.py` green unchanged;
   `python3 -m examples.run_agentic_search --help` runs.

## Task 3 — widen the source scan

1. `test_run_agentic_search_has_no_minisweagent_dependency` scans the package,
   not one file, and asserts it found at least four sources.
   → verify: mutation — add `minisweagent` to `routing.py`, test goes red.

## Task 4 — verification

1. `pytest` (full default suite).
2. `python3 -m examples.run_agentic_search --help`, and the bamboogle scripts
   that import from this module.
3. `ruff check . --fix && ruff format .`
