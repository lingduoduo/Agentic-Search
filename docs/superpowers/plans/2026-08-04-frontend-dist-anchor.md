# Frontend Dist Anchor Implementation Plan

Spec: `docs/superpowers/specs/2026-08-04-frontend-dist-anchor-design.md`

## Global Constraints

- One-line production change. The mount and SPA handler are not touched.
- Tests must not depend on `web/dist` existing: it is gitignored and absent in CI.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/servers/web/app.py` | modify | correct the parent count |
| `tests/unit/servers/web/test_frontend_dist_path.py` | new | the bug + two controls |

---

### Task 1: The bundle is found where it is built

- [x] **Step 1: Write the failing test**

`_fake_checkout` builds a tree shaped like this repository in `tmp_path` and the
test points `web_app.__file__` at the fake `app.py`, so the real parent
arithmetic runs against a controlled tree. This is why no `npm run build` is
needed.

Three tests: a bundle at the fabricated root is found; no bundle returns `None`;
`index.html` without `assets/` returns `None`.

- [x] **Step 2: Run to verify it fails**

Verify: `AssertionError: assert None == …/web/dist` — the bug, not a missing
helper. The two controls pass already, which is what makes them controls.

- [x] **Step 3: Implement**

`parents[3]` → `parents[4]`, plus a comment counting the levels, because the
failure mode is silent and the next person moving this file must redo the count.

- [x] **Step 4: Run to verify it passes** — 3 passed.

---

### Task 2: Prove it end to end

- [x] Start the backend and check what is actually served.

Verify: `/` returns the built `index.html` — the hashed asset names in the
response match `web/dist/index.html` exactly; `/assets/<hashed>.js` → 200; SPA
routes `/search` and `/tools` → 200; `/health` → 200.

---

### Task 3: Suite and lint

- [x] `python3 -m pytest` — 2890 passed.
- [x] `ruff check . && ruff format --check .`

---

## Self-Review

**Why the fabricated tree instead of the real one.** A test asserting against the
real checkout would pass or fail depending on whether the developer had run
`npm run build`, and would be vacuous in CI where `web/dist` never exists. The
fake tree makes the arithmetic itself the subject.

**Why `__file__` monkeypatching is legitimate here.** The function's input *is*
its own module path; patching it is patching the input, not adding a test-only
parameter to production code.

**Scope.** One line and a comment. The `StaticFiles` mount and SPA handler needed
no change — they were correct all along, just unreachable.
