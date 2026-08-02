# Surface Generation Truncation Implementation Plan

**Goal:** A truncated answer says so — from the server manager that detects it through to the UI — instead of appearing as a complete but nonsensical fragment.

**Architecture:** The manager records truncation keyed by `request_id`; the agent loop pops it and puts it on `AgentLoopOutput`; the tool API returns it; the tool view renders an advisory notice naming the env var that fixes it.

**Tech Stack:** Python 3.12, FastAPI, transformers, pytest; React 19 + Vitest.

**Spec:** `docs/superpowers/specs/2026-08-01-surface-generation-truncation-design.md`

## Global Constraints

- Work on branch `feat/surface-generation-truncation`. Never commit to `main`.
- No Protocol change: managers without the hook must keep working untouched.
- No change to when truncation happens, or to the default timeout.
- `python3 -m pytest`, `npm run typecheck`, `npx vitest run`, and ruff all pass.

## Tasks

- [x] **Task 1 — Record it.** `_record_truncation` / `pop_truncated` on
      `LocalServerManager`, keyed by `request_id`, bounded by
      `_TRUNCATION_RECORD_LIMIT`.
      *Verify:* pops once then False; keyed per request; bounded under flood.

- [x] **Task 2 — Carry it.** `AgentLoopOutput.truncated`;
      `generate_response_ids` pops via `getattr` so hookless managers are
      unaffected; `ToolAgentLoop.run` fills the field.
      *Verify:* truthy manager → `truncated=True`; falsy → False; manager with
      no hook → False, no error.

- [x] **Task 3 — Return it.** `extra["truncated"]` from `_run_tool_agent`;
      `ToolAgentMessageResponse.truncated`; `truncated` in the SSE `done` event.
      *Verify:* model default False; the endpoint reports True.

- [x] **Task 4 — Show it.** `ToolStreamEvent.done.truncated`; a `role="status"`
      notice in `ToolAgentView` naming `AGENTIC_SEARCH_GENERATION_TIMEOUT`;
      advisory styling.
      *Verify:* notice on truncated runs, absent on complete ones.

- [x] **Task 5 — Test the real call path.** A test that drives `generate()`
      against a stub *model* rather than a stub *manager*.
      *Verify:* reintroducing the `_generate_sync` signature break fails this
      test while the mocked ones pass.

- [x] **Task 6 — Verify against the real model** with the timeout forced low.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Backend | `python3 -m pytest` | 2821 passed |
| Frontend | `npm run typecheck && npx vitest run` | clean, 167 passed |
| Lint | `ruff check . && ruff format .` | clean |
| End-to-end | live model, timeout 20s | `truncated=True`, 86 tokens, cut mid-sentence |
