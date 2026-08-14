# Plan — cap the openai test dependency below 3.0.0

Spec: [2026-08-14-openai-major-pin-design.md](../specs/2026-08-14-openai-major-pin-design.md)

## Task 1 — reproduce CI in a clean venv

The developer environment resolves openai 2.x and passes, so the bug is
invisible locally. Build a venv from `requirements-unit-test.txt` + `pip
install -e .` and run `tests/unit/test_embedding_cache.py`.

**Verify:** 6 failures, matching the names in the CI log, with
`AttributeError: module aiohttp has no attribute SocketTimeoutError`.
Confirm `pip list` shows `openai 3.0.0` against `aiohttp 3.9.3`.

*Done — 6 failed, 11 passed; openai 3.0.0, aiohttp 3.9.3.*

## Task 2 — apply the cap

Change `openai>=1.0.0` to `openai>=1.0.0,<3` in `requirements-unit-test.txt`,
with a comment recording the incompatibility, why the aiohttp pin was not
raised instead, and that the cap should be lifted deliberately with aiohttp.

**Verify:** re-resolve the same venv; `pip list` shows openai 2.x, and the
embedding-cache tests pass.

*Done — openai 2.54.0, 17 passed.*

## Task 3 — full suite, CI-faithful

Per-slice runs have missed regressions in this repo before. Run all of
`tests/unit/` in the clean (torch-free) venv, with the ambient API keys unset
so a developer `.env` cannot mask an LLM-dependent failure.

**Verify:** no failures, and the pass+skip total matches CI's collected count.

*Done — 2648 passed, 36 skipped, matching CI's 2684 collected.*

## Task 4 — ship

Own branch off `main` (this is a separate deliverable from the intent work),
spec + plan committed alongside the change, PR opened.

**Verify:** the `Python Unit Tests` check is green on the PR — the first time
it has been on this repo since #510.
