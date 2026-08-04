# Verify-Script ACL Isolation Implementation Plan

Spec: `docs/superpowers/specs/2026-08-04-verify-script-acl-isolation-design.md`

## Global Constraints

- No change to any enforcement path. `app.py`, `routing_tools.py` and
  `search.py` are not touched by this plan.
- `--ignore-acl` defaults off; every existing ACL test must keep passing
  unchanged.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/servers/retrieval/demo.py` | modify | opt-out flag |
| `examples/verify_identity_capabilities.sh` | modify | use it; report curl failure |
| `tests/unit/test_retrieval_server_acl.py` | modify | cover the flag |

---

### Task 1: `demo.py` can be told to ignore ACLs

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_retrieval_server_acl.py`, reusing the existing `DOCS` and
`_ids` helpers:

- `test_ignore_acl_serves_restricted_documents_anyway` — `create_app(..., ignore_acl=True)`
  returns `theirs` for a request filtered to `access_acl: ["public"]`.
- `test_ignore_acl_is_reachable_from_the_command_line` — `parse_args()` under a
  patched `sys.argv` yields `ignore_acl` True with the flag and False without.

The second exists because a flag parsed but never passed to `create_app` fails
silently in exactly the way this work is about.

- [x] **Step 2: Run to verify they fail**

Expected: `TypeError` on the unexpected keyword, and argparse `SystemExit: 2`
on `unrecognized arguments: --ignore-acl`.

- [x] **Step 3: Implement**

`create_app(retriever, *, ignore_acl: bool = False)` resolving
`filters = None if ignore_acl else body.filters`; an `--ignore-acl` store_true
argument; `main()` passing it through.

- [x] **Step 4: Run to verify they pass, including the pre-existing ACL tests**

Verify: `pytest tests/unit/test_retrieval_server_acl.py` — 7 passed.

---

### Task 2: The script isolates the web layer again

- [x] **Step 1: Start `demo.py` with `--ignore-acl` and correct the header**

The header must say why the flag is there, not merely that it is.

- [x] **Step 2: Run the script**

Verify: `PASS: neither identity can read another user's document`.

- [x] **Step 3: Prove the check can fail**

Stub `_enforce_access` to `return documents`, re-run, confirm FAIL, restore, and
confirm `git diff` on `app.py` is empty.

Verify: `FAIL: a restricted document leaked`, with both identities reported True.

---

### Task 3: The tool leg completes or says why not

- [x] **Step 1: Make a curl failure non-fatal and reported**

`|| true` on the `tool_leg` curl; a `000` check raising a FAIL that names the
timeout as the likely cause, placed before the existing 400 checks.

- [x] **Step 2: Confirm the silent-abort mechanism**

Verify: a `set -euo pipefail` script assigning `$(curl …)` from a dead port exits
non-zero with no output — the behaviour being fixed.

---

### Task 4: Run the tool-agent leg for real

- [x] **Step 1: Pass `SEARCH_AGENT_MODEL` through**

- [x] **Step 2: Run the whole script with a local model**

Verify: both legs PASS, including
`PASS (model-dependent): the tool agent did not surface another user's document`.

- [x] **Step 3: Prove the tool's enforcement is what withholds the document**

Against `demo.py --ignore-acl`, call `build_search_routing_tool` twice — with
`SearchFilters(access_acl=["public"])` and with `filters=None` — and compare.
Uses the builder's own documented API rather than deleting the enforcement
block, so no production code is mutated to run the test.

Verify: with filters, one document and no `confidential`; without, two documents
including `confidential`.

---

### Task 5: Full suite and lint

- [x] `python3 -m pytest`
- [x] `ruff check . && ruff format --check .`

---

## Self-Review

**Spec coverage.** `--ignore-acl` → Task 1. Script uses it → Task 2.
Pass-through → Task 4 Step 1. Curl failure reported → Task 3. Every Verification
bullet in the spec maps to a step that names its expected output.

**Where this plan deviates from its own instruction.** Task 4 Step 3 was
specified as "remove the enforcement block and re-run". That edit was blocked as
an unsafe modification, and the substitute — exercising the builder with and
without `filters` — is strictly better: it produces the same counterfactual from
the public API, and leaves no window in which the working tree carries a
disabled access control.

**Scope held.** No enforcement path is touched. The one production change is a
default-off flag on the demo server.
