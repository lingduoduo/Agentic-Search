# Curate Session Ownership Implementation Plan

Spec: `docs/superpowers/specs/2026-08-04-curate-session-ownership-design.md`

## Global Constraints

- One production change, in `_gather_sources`. No router or MCP-tool change:
  the point is that both inherit the check.
- The existing curation tests must pass untouched — a fix that needed them
  rewritten would mean the behaviour changed for owners too.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/memory/service.py` | modify | ownership predicate at the read |
| `tests/unit/memory/test_curate_session_ownership.py` | new | the leak + two controls |

---

### Task 1: A session you may not read is not read

- [x] **Step 1: Write the failing test**

`test_another_users_session_is_not_read`: u2 owns a session containing a secret;
u1 curates with that `session_id`.

The LLM double raises `AssertionError` on any `stream()` call. Asserting only on
the returned status would pass for the wrong reason — the transcript could still
have reached the prompt while the model happened to write nothing. The double
makes "the prompt was built at all" the failure.

Two controls in the same file, which must pass both before and after:

- `test_the_callers_own_session_is_still_read` — the owner still gets the
  transcript, so the fix withholds rather than blanket-refusing.
- `test_a_session_with_no_owner_stays_readable` — NULL-owner sessions stay
  readable, the anonymous path.

- [x] **Step 2: Run to verify it fails**

Verify: `AssertionError: the LLM was handed a transcript it may not read`, with
the two controls already passing.

- [x] **Step 3: Implement**

Add `_readable(session, user_id)` — `session is not None and session.user_id in
(None, user_id)` — and filter the by-id branch through it.

- [x] **Step 4: Run to verify it passes**

Verify: `pytest tests/unit/memory/` — 22 passed, the pre-existing curation tests
unmodified.

---

### Task 2: Suite and lint

- [x] `python3 -m pytest`
- [x] `ruff check . && ruff format --check .`

---

## Self-Review

**Spec coverage.** Ownership predicate → Task 1 Step 3. All three Verification
bullets → the three tests in Task 1 Step 1.

**Why the LLM double is the assertion.** The spec's guarantee is about what
enters the prompt, not about what comes back. A status-only assertion would hold
even if the leak were intact, which is the same class of false pass #487 shipped.

**Scope.** The shared `default_user` bucket is untouched and recorded as a
Non-goal; it is a real gap, but a different decision.
