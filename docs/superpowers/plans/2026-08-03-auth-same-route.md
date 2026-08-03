# Auth Same-Route Implementation Plan

**Goal:** Signing in narrows results to what the user may see; it does not change which route runs.

**Architecture:** Carry the document ACL through `SearchPage`, enforce it on the route itself (not by trusting the retrieval server), then delete the divert that sent every filtered query to `_auto_search_pipeline`.

**Spec:** `docs/superpowers/specs/2026-08-03-auth-same-route-design.md`

## Global Constraints

- Work on branch `fix/auth-same-route`. Never commit to `main`.
- Enforcement must not depend on the retrieval server: `demo.py` and `hybrid.py` ignore `filters`.
- Anonymous behaviour unchanged.
- `python3 -m pytest` and `ruff check . && ruff format .` pass before commit.

## Tasks

- [x] **Task 1 — Rewrite the routing tests** to assert the intended contract
      (same route signed in or not, filters threaded everywhere).
      *Verify:* they fail against the divert.

- [x] **Task 2 — Carry the ACL.** `SearchPage.metadata` from
      `SearchResult.metadata`; merged into `ContextDocument` under the labels.
      *Verify:* a document's `acl` reaches the caller.

- [x] **Task 3 — Enforce on the route.** `_enforce_access` on the direct path
      before the gate and on escalated documents.
      *Verify:* another user's document never reaches the caller; unfiltered
      requests keep everything.

- [x] **Task 4 — Remove the divert.**
      *Verify:* signed-in and anonymous produce the same `search_mode`.

- [x] **Task 5 — Prove it end to end** against `demo.py`, which ignores filters.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Unit + regression | `python3 -m pytest` | 2830 passed |
| Lint | `ruff check . && ruff format .` | clean |
| Live routing | `/api/agent` "RAG" anon vs signed in | both `direct`/`semantic`/2 docs; 8.17s → 0.027s |
| Live ACL | restricted doc via demo.py | anonymous sees it, signed-in user does not |
