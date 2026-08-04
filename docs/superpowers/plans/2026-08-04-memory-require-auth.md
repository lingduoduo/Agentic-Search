# Memory Require-Auth Implementation Plan

Spec: `docs/superpowers/specs/2026-08-04-memory-require-auth-design.md`

## Global Constraints

- Default behaviour must not change. Every existing memory test passes unmodified.
- The flag governs both doors — the web router and the MCP tools — or not at all.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/memory/router.py` | modify | `require_auth` → 401 |
| `src/internal/servers/web/app.py` | modify | settings field + plumbing |
| `src/internal/mcp_server/tools/memory.py` | modify | same var, separate process |
| `tests/unit/servers/web/test_memory_router.py` | modify | strict + default control |
| `tests/unit/test_mcp_memory_tools.py` | modify | strict + token control |
| `docs/{configuration,cli,mcp}.md` | modify | document the switch |

---

### Task 1: The router can require authentication

- [x] **Step 1: Write the failing tests**

- `test_strict_mode_refuses_anonymous_callers` — all five read/write routes 401.
- `test_strict_mode_still_serves_an_authenticated_caller_their_own_bucket` — the
  control; a pre-seeded `default_user` memory must not appear for `alice`.
- `test_default_mode_leaves_the_anonymous_bucket_reachable` — pins the July
  research-use ruling so a later change cannot erode it silently.

- [x] **Step 2: Run to verify they fail**

Verify: `TypeError: create_memory_router() got an unexpected keyword argument
'require_auth'` on the two strict tests; the default-mode control passes already,
which is the point of writing it.

- [x] **Step 3: Implement**

`require_auth: bool = False`; `_uid` raises `HTTPException(401)` when the caller
is anonymous and the flag is set.

- [x] **Step 4: Run to verify they pass** — 6 passed.

---

### Task 2: Plumb the flag

- [x] `ServiceSettings.memory_require_auth`, `_flag("AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH")`,
  `_register_routers(..., memory_require_auth=...)`, passed at the call site.

---

### Task 3: The MCP door honours the same flag

- [x] **Step 1: Implement** — `_resolve_user_id` raises `PermissionError` when the
  var is set and no token resolved. Read via `os.getenv` at call time: the MCP
  server is a separate process and never builds a `ServiceSettings`.

- [x] **Step 2: Test, then mutation-check the test**

Written after the implementation, so the tests passed on the first run and had
proved nothing. Verified by mutation instead: repointing the `os.getenv` lookup
at an unset variable name.

Verify: `Failed: DID NOT RAISE <class 'PermissionError'>` — the test does catch
the missing enforcement. Probe reverted; `git diff` confirms.

---

### Task 4: Docs, suite, lint

- [x] `configuration.md` env row; `cli.md` names the sharing the flag fixes;
  `mcp.md` states the two-door coupling.
- [x] `python3 -m pytest`
- [x] `ruff check . && ruff format --check .`

---

## Self-Review

**Spec coverage.** Router switch → Task 1. Plumbing → Task 2. Both doors →
Task 3. All four Verification bullets map to a named test.

**Where this plan departs from TDD, stated rather than hidden.** Task 3's
implementation preceded its tests. Rather than claim a red-green cycle that did
not happen, the tests were mutation-checked — weaker than watching them fail
first, and recorded as such.

**Default-preservation is a test, not a claim.** `test_default_mode_leaves_the_anonymous_bucket_reachable`
exists so "off is unchanged" is enforced rather than asserted in a PR body.
