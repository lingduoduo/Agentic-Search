# Generated Context Pack

# Api Request Routing Docs

## Sources

- [Specification: 2026-07-13-api-request-routing-docs-design.md](../specs/2026-07-13-api-request-routing-docs-design.md)
- [Plan: 2026-07-13-api-request-routing-docs.md](../plans/2026-07-13-api-request-routing-docs.md)

## Specification Context

### Goal

Make the maintained documentation describe API request routing accurately and consistently, including the search-provider precedence introduced in PR #410.

### Scope

Create `docs/request-routing.md` as the canonical routing contract. Update `README.md` and every maintained top-level guide under `docs/` wherever routing affects that guide's subject. Do not rewrite historical files under `docs/superpowers/specs/` or `docs/superpowers/plans/`; this design document is the only new file in that historical area.

### Verification

- Search all maintained documentation for stale routing statements and reconcile each hit.
- Validate relative Markdown links.
- Run the routing and web fallback unit suites because the docs include executable examples and exact response semantics.
- Run formatting or documentation checks available in the repository and `git diff --check`.

## Implementation Plan Context

### Global Constraints

- Historical `docs/superpowers/specs/` and `docs/superpowers/plans/` files remain unchanged except for this feature's spec and plan.
- Auto-routed search with `source_provider=auto` is documented as internal retrieval → sufficiency gate → SerpAPI → configured browser-search service → deterministic no-evidence response.
- Auto-routed search never substitutes a local-model internal-knowledge answer for missing evidence.
- Explicit request modes retain their separate documented behavior.
- Browser search is an HTTP service backed by `playwright-cli`, not in-process browser automation in the request handler.

---

### Task 1: Establish the canonical routing contract

**Files:**
- Create: `docs/request-routing.md`
- Inspect: `src/internal/servers/web/app.py`
- Inspect: `src/internal/servers/web/intent_routing.py`
- Inspect: `src/internal/servers/web/models.py`

**Interfaces:**
- Consumes: `/api/agent` and `/api/agent/stream` request handling, `RouteStrategy`, direct-search sufficiency gate, and provider adapters.
- Produces: canonical terminology and links consumed by every documentation update in Task 2.

- [ ] **Step 1: Extract exact request fields, modes, response metadata, router precedence, and provider fallbacks from the implementation**

Run:

Expected: definitions for the API model, dispatcher, three-way router, and sequential search fallback.

…

### Task 2: Synchronize all maintained documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/api-reference.md`
- Modify: `docs/architecture.md`
- Modify: `docs/configuration.md`
- Modify: `docs/retrieval.md`
- Modify: `docs/frontend.md`
- Modify: `docs/testing.md`
- Modify: `docs/training-and-evaluation.md`
- Modify: `docs/mcp.md`

**Interfaces:**
- Consumes: canonical terms and anchors from `docs/request-routing.md`.
- Produces: concise, audience-specific routing summaries with links back to the canonical contract.

- [ ] **Step 1: Add the routing entry point to the README**

…

### Task 3: Verify documentation and routing behavior

**Files:**
- Verify: `README.md`
- Verify: `docs/*.md`
- Test: `tests/unit/test_execution_fallbacks.py`
- Test: `tests/unit/servers/web/test_agent_router.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: completed routing guide and synchronized summaries.
- Produces: evidence that links, examples, and documented behavior match the test-backed implementation.

- [ ] **Step 1: Validate local Markdown links**

Run the repository's documentation/link checker if present; otherwise enumerate Markdown links in `README.md` and `docs/*.md` and verify every relative target exists.

Expected: zero missing local targets.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
