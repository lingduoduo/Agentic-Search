# Curate Strict Session Ownership Implementation Plan

Spec: `docs/superpowers/specs/2026-08-04-curate-strict-session-ownership-design.md`

## Global Constraints

- Two production edits, both in `memory/service.py`. No router or MCP change.
- The capability removed must be pinned by a test, not described in a PR body.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/memory/service.py` | modify | strict predicate + specific message |
| `tests/unit/memory/test_curate_session_ownership.py` | modify | invert ownerless, pin the loss |
| `docs/cli.md` | modify | correct the #496 line |

---

### Task 1: Ownership is strict

- [x] **Step 1: Invert the ownerless test, add the message test**

`test_a_session_with_no_owner_is_not_read_either` replaces
`test_a_session_with_no_owner_stays_readable`. The two #494 tests (another
user's, and the caller's own) stay untouched as controls — a change that needed
them rewritten would mean the fix broke owners too.

- [x] **Step 2: Run to verify they fail**

Verify: `AssertionError: the LLM was handed a transcript it may not read` on
both new tests — the ownerless transcript still reaching the prompt.

- [x] **Step 3: Implement** — `session.user_id == user_id`, and a
`session_id`-specific empty message.

- [x] **Step 4: Run to verify** — `tests/unit/memory/` 24 passed.

---

### Task 2: Measure the real cost before documenting it

- [x] **Step 1: Check what an anonymous caller loses**

Verify: `list_sessions_for_user("default_user")` returns `[]` for a NULL-owned
session. The no-flag path never reached anonymous sessions, so `-session-id` was
the only route — the loss is *all* conversation curation for anonymous callers,
not just the by-id route.

This contradicted the first draft of the doc note, which claimed plain
`memory curate` still worked. Written from the check, not from the assumption.

- [x] **Step 2: Pin it** —
`test_an_anonymous_caller_can_no_longer_curate_from_conversations` asserts both
halves, so the scope of the loss cannot be rediscovered later as a surprise.

- [x] **Step 3: Correct `docs/cli.md`** — the #496 line said "one you own, or one
with no owner", now false.

---

### Task 3: Suite and lint

- [x] `python3 -m pytest` — 2898 passed.
- [x] `ruff check . && ruff format --check .`

---

## Self-Review

**This reverses a decision I recommended keeping**, at the user's explicit
direction after I stated the objection. The objection was that strict ownership
removes a capability invisibly; the mitigation is Task 2 — the loss is measured,
reported in the response, pinned by a test, and called out in the docs. That
addresses the objection rather than dismissing it.

**Where the plan corrected itself.** Task 2 Step 1 exists because the doc note I
first wrote was wrong: I assumed the no-flag path still served anonymous callers.
The check said otherwise. The step is kept in the plan so the next reader sees
the claim was measured.

**Scope.** The message change is not scope creep: it is the mitigation for the
one risk the change carries, and reuses the existing `empty` status rather than
adding an error path.
