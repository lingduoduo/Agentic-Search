# API Request Routing Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one authoritative API request-routing guide and align every maintained top-level document with the current internal → SerpAPI → browser provider order.

**Architecture:** `docs/request-routing.md` owns the complete behavioral contract. Other maintained guides contain only the details relevant to their audience and link to the canonical guide, preventing duplicated routing prose from drifting.

**Tech Stack:** Markdown, FastAPI request/response models, pytest routing tests, shell-based link and stale-text checks.

## Global Constraints

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

```bash
rg -n "class AgentRequest|async def _run_agent_impl|async def _run_auto_routed|async def _run_search_direct_or_escalate|def route_query|class RouteStrategy" src/internal/servers/web
```

Expected: definitions for the API model, dispatcher, three-way router, and sequential search fallback.

- [ ] **Step 2: Write the complete guide**

Create `docs/request-routing.md` with these concrete sections: endpoints and shared dispatcher; request fields; explicit mode table; auto-router decision precedence; search-provider sequence; sufficiency gate; filter behavior; response and SSE fields; configuration dependencies; `RAG`/`GRPO` walkthroughs; troubleshooting; code ownership.

- [ ] **Step 3: Check the guide against the implementation**

Run:

```bash
rg -n "internal retrieval|SerpAPI|browser|No results|No sources|local model|source_provider|route_degraded|search_mode" docs/request-routing.md src/internal/servers/web/app.py
```

Expected: every behavioral claim in the guide has a corresponding implementation term or branch.

- [ ] **Step 4: Commit the canonical guide**

```bash
git add docs/request-routing.md
git commit -m "docs: add API request routing guide"
```

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

Add a concise request-routing paragraph and a Documentation-list entry. Replace troubleshooting text that suggests omitting `SEARCH_AGENT_MODEL` is the auto-route provider fallback.

- [ ] **Step 2: Replace stale API and architecture behavior**

In `docs/api-reference.md`, document exact request and response fields, explicit modes, sequential provider precedence, SSE metadata, and examples. Remove the obsolete “known gap” claiming local-model auto routes cannot reach web search.

In `docs/architecture.md`, add an end-to-end request flow and clearly separate strategy routing, source-provider selection, and retrieval-backend routing.

- [ ] **Step 3: Align configuration and retrieval guides**

In `docs/configuration.md`, explain which variables enable SerpAPI, browser fallback, intent models, and explicit local policy modes, including precedence.

In `docs/retrieval.md`, document internal sufficiency gating and sequential external fallback, including filter constraints and deterministic empty/unreachable results.

- [ ] **Step 4: Update adjacent guides**

In `docs/frontend.md`, connect `intent`, route metadata, SSE events, source cards, and the inspector to routing.

In `docs/testing.md`, add focused routing and fallback test commands.

In `docs/training-and-evaluation.md`, state that normal API handling performs inference only and that explicit trainable agent modes do not train during requests.

In `docs/mcp.md`, distinguish MCP tool calls from the web API auto-router.

- [ ] **Step 5: Scan maintained docs for stale claims**

Run:

```bash
rg -n -i "web reachability|internal-corpus only|web is reached only|fan-out|parallel|source_provider=auto|search_agent|route_degraded|SerpAPI|browser" README.md docs/*.md
```

Expected: remaining matches use current terminology; no maintained guide claims that auto search answers from local knowledge after empty retrieval or that auto providers always fan out in parallel.

- [ ] **Step 6: Commit synchronized guides**

```bash
git add README.md docs/api-reference.md docs/architecture.md docs/configuration.md docs/retrieval.md docs/frontend.md docs/testing.md docs/training-and-evaluation.md docs/mcp.md
git commit -m "docs: align guides with request routing"
```

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

- [ ] **Step 2: Run focused behavioral tests**

```bash
pytest -q tests/unit/test_execution_fallbacks.py tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_web_experience_app.py
```

Expected: all collected tests pass.

- [ ] **Step 3: Run whitespace and final diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended documentation changes remain before the final commit.

- [ ] **Step 4: Push and update PR #410**

```bash
git push
gh pr edit 410 --repo lingduoduo/Agentic-Search-GRPO --title "fix: prioritize search providers before local model" --body-file /private/tmp/pr410-body.md
```

Expected: PR #410 contains the routing implementation, canonical guide, synchronized documentation, and verification summary.
