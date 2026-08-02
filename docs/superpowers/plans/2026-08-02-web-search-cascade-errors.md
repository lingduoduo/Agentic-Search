# web_search Cascade Errors Implementation Plan

**Goal:** When `web_search` cannot run, say why instead of returning "No results found."

**Architecture:** `make_web_cascade_search` accumulates each leg's failures and returns them when no leg is usable. `format_search_pages` already renders error pages, so nothing downstream changes.

**Tech Stack:** Python 3.12, pytest.

## Global Constraints

- Work on branch `fix/web-search-cascade-errors`. Never commit to `main`.
- A genuinely empty search must still report empty — no invented failures.
- No secrets in surfaced errors (`_redact_secret_params` already applies).
- `python3 -m pytest` and `ruff check . && ruff format .` pass before commit.

## Tasks

- [x] **Task 1 — Accumulate failures.** Collect error pages from the SerpAPI leg,
      and error pages or the raised exception from the browser leg; return them
      instead of `[]`.
      *Verify:* a 429 from SerpAPI surfaces "Too Many Requests"; both legs
      failing surfaces both; a browser exception surfaces its message.

- [x] **Task 2 — Name the missing fallback**, but only when a leg errored.
      *Verify:* the error mentions `AGENTIC_SEARCH_BROWSER_SEARCH_URL`; a
      genuinely empty search still returns `[]`.

- [x] **Task 3 — Verify against the live stack** with the real (rate-limited)
      key.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit + regression | `python3 -m pytest` | 2824 passed (4 new) |
| Lint | `ruff check . && ruff format .` | clean |
| Live | `POST /admin/tools/web_search/invoke` | reports the 429 and the missing fallback, key redacted |
