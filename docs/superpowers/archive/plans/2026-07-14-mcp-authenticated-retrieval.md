# Authenticated MCP Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every MCP indexed-document read through the authenticated web backend so existing user ACLs and requested document-set restrictions apply before evidence reaches an MCP client or LLM.

**Architecture:** Harden the existing non-streaming `/search/send-search-message` path so it derives ACLs from the authenticated request instead of trusting caller-supplied ACL values. Add a focused MCP HTTP client that forwards the request-local bearer token to that endpoint, then migrate all three private retrieval tools to the client while preserving their MCP result contracts.

**Tech Stack:** Python 3.12, FastAPI, FastMCP, HTTPX, Pydantic, pytest, Ruff

## Global Constraints

- The web backend is the single authority for user, group, connector, and document-set access.
- Missing or rejected authentication must never fall back to the raw retrieval service.
- Existing MCP tool names and successful response fields remain stable.
- `document_set_names=None` and `[]` mean no caller-specified document-set restriction; normal ACLs still apply.
- Public web search, URL fetching, query expansion, and non-MCP raw retrieval callers remain unchanged.
- Use strict red-green-refactor TDD for every behavior change.

---

### Task 1: Enforce authenticated filters in the existing web search endpoint

**Files:**
- Modify: `src/internal/servers/query_and_chat/search_backend.py`
- Create: `tests/unit/servers/query_and_chat/test_search_backend.py`

**Interfaces:**
- Consumes: `user_from_headers(headers) -> AuthenticatedUser | None`, `build_user_only_filters(principal_id, email=..., group_ids=...) -> SearchFilters`
- Produces: `_authenticated_search_filters(request: Request, requested: SearchFilters | None) -> SearchFilters`; non-streaming `POST /search/send-search-message` applies server-derived `access_acl` and caller narrowing fields.

- [ ] **Step 1: Write failing endpoint tests**

Add tests that patch `run_expanded_search`, send authenticated headers, and assert the received filter combines the server-built ACL with caller `source_types`, `document_sets`, `tags`, and `time_cutoff`. Add a test proving a caller-supplied `access_acl` is discarded, plus a `401` test for missing authentication.

```python
assert captured.filters.access_acl == ["public", "user:alice"]
assert captured.filters.document_sets == ["engineering"]
assert "user:other" not in captured.filters.access_acl
assert response.status_code == 401
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/unit/servers/query_and_chat/test_search_backend.py -q`

Expected: failures show the endpoint currently accepts caller ACLs and permits an unauthenticated request.

- [ ] **Step 3: Implement the authenticated filter merge**

Add a helper that requires a non-anonymous user and preserves only safe caller narrowing fields:

```python
def _authenticated_search_filters(
    request: Request,
    requested: SearchFilters | None,
) -> SearchFilters:
    user = user_from_headers(request.headers)
    if user is None or user.is_anonymous:
        raise HTTPException(status_code=401, detail="Authentication required.")
    acl = build_user_only_filters(
        user.id,
        email=user.email,
        group_ids=user.group_ids,
    )
    return SearchFilters(
        source_types=requested.source_types if requested else None,
        document_sets=requested.document_sets if requested else None,
        tags=requested.tags if requested else None,
        access_acl=acl.access_acl,
        time_cutoff=requested.time_cutoff if requested else None,
    )
```

Resolve this filter before both JSON and streaming execution and pass it to `run_expanded_search`.

- [ ] **Step 4: Run focused and adjacent tests**

Run: `pytest tests/unit/servers/query_and_chat/test_search_backend.py tests/unit/servers/web/test_web_experience_app.py -q`

Expected: all pass, with no response-model changes.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/query_and_chat/search_backend.py tests/unit/servers/query_and_chat/test_search_backend.py
git commit -m "fix: enforce ACLs on web search endpoint"
```

### Task 2: Add a shared authenticated MCP retrieval client

**Files:**
- Create: `src/internal/mcp_server/retrieval_client.py`
- Modify: `src/internal/mcp_server/utils.py`
- Test: `tests/unit/test_mcp_retrieval_client.py`

**Interfaces:**
- Consumes: `require_access_token() -> AccessToken`, `build_web_base_url() -> str`, `get_http_client() -> httpx.AsyncClient`
- Produces: `authenticated_retrieve(query: str, *, top_k: int, document_set_names: list[str] | None = None) -> list[AuthenticatedDocument]`
- Produces: `AuthenticatedDocument(title: str, url: str | None, content: str, score: float, metadata: dict[str, Any])`

- [ ] **Step 1: Write failing client tests**

Cover token forwarding, payload normalization, successful response parsing, `401`/`403`, `5xx`, malformed responses, and empty results. Patch `require_access_token` and the shared HTTP client; assert no raw retrieval helper is imported or called.

```python
client.post.assert_awaited_once_with(
    f"{base_url}/search/send-search-message",
    headers={"Authorization": "Bearer secret"},
    json={
        "search_query": "GRPO",
        "filters": {"document_sets": ["ml"]},
        "run_query_expansion": False,
        "num_hits": 5,
        "stream": False,
    },
)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/unit/test_mcp_retrieval_client.py -q`

Expected: collection fails because `retrieval_client.py` does not exist.

- [ ] **Step 3: Implement the focused client**

Use an immutable dataclass for normalized documents. Normalize `document_set_names or None`; call `response.raise_for_status()`; translate authentication, authorization, timeout, transport, server, and schema failures into a dedicated `AuthenticatedRetrievalError` whose message is safe for MCP output.

```python
token = require_access_token()
headers = {"Authorization": f"Bearer {token.token}"}
filters = {"document_sets": document_set_names} if document_set_names else None
```

Never catch an error and retry against `AGENTIC_SEARCH_RETRIEVAL_PORT`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/unit/test_mcp_retrieval_client.py tests/unit/test_mcp_server.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/internal/mcp_server/retrieval_client.py tests/unit/test_mcp_retrieval_client.py
git commit -m "feat: add authenticated MCP retrieval client"
```

### Task 3: Migrate raw MCP search and retrieval tools

**Files:**
- Modify: `src/internal/mcp_server/tools/search.py`
- Modify: `src/internal/mcp_server/tools/research.py`
- Test: `tests/unit/test_mcp_server.py`

**Interfaces:**
- Consumes: `authenticated_retrieve(query, top_k=..., document_set_names=...)`
- Produces: unchanged `search_indexed_documents` result shape and unchanged `retrieve_documents` result shape.

- [ ] **Step 1: Replace mocks with failing authenticated-client expectations**

For `search_indexed_documents`, assert query, `top_k=5`, and normalized document-set names reach `authenticated_retrieve`. For `retrieve_documents`, assert query and caller `top_k` are forwarded. Add missing-token and backend-authorization tests proving neither function calls `retrieval_search` or `retrieve_context`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/unit/test_mcp_server.py -q`

Expected: failures show both tools still call raw retrieval helpers.

- [ ] **Step 3: Migrate both tools**

Remove `_retrieval_url`, `retrieval_search`, and `retrieve_context` dependencies from these private paths. Format `AuthenticatedDocument` values into the existing payloads. Treat an authenticated empty document list as a successful empty result; reserve `error` for client/backend failures.

Update the `search_indexed_documents` docstring to state that `document_set_names` narrows authorized results. Continue accepting `source_types`, `time_cutoff`, and `skip_query_expansion` for compatibility without expanding this task's scope.

- [ ] **Step 4: Run MCP and registry regressions**

Run: `pytest tests/unit/test_mcp_server.py tests/unit/test_mcp_retrieval_client.py tests/unit/test_tool_registry.py tests/unit/test_tool_arg_validation.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/internal/mcp_server/tools/search.py src/internal/mcp_server/tools/research.py tests/unit/test_mcp_server.py
git commit -m "fix: authenticate MCP document retrieval"
```

### Task 4: Ground MCP answers only in authenticated evidence

**Files:**
- Modify: `src/internal/mcp_server/tools/chat.py`
- Test: `tests/unit/test_mcp_server.py`

**Interfaces:**
- Consumes: `authenticated_retrieve(question, top_k=top_k)`
- Produces: unchanged `{answer, citations, sources}` success shape from `ask_agentic_search`.

- [ ] **Step 1: Write failing grounding tests**

Patch `authenticated_retrieve` to return two authorized documents and assert the answer path receives only those documents. Add empty-evidence, missing-token, `403`, LLM-enabled, and extractive-fallback cases. Assert `answer_with_retrieval` is never called because it would perform a second unauthenticated retrieval.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/unit/test_mcp_server.py -q`

Expected: failures show `ask_agentic_search` still calls `answer_with_retrieval` with a raw retrieval URL.

- [ ] **Step 3: Split retrieval from synthesis**

Use `authenticated_retrieve` to obtain evidence, convert it to `SearchResult` values, build a `SearchContextBundle` with `build_context_bundle`, and call the existing `generate_answer(AnswerRequest(...), llm=llm)` synthesis-only path. This shares generation and grounding verification without calling `answer_with_retrieval` or performing a second retrieval.

For no authorized evidence, return the existing safe no-evidence answer contract with empty citations and sources. Never invoke the LLM without evidence.

- [ ] **Step 4: Run context and MCP tests**

Run: `pytest tests/unit/test_mcp_server.py tests/unit/test_context_pipeline.py tests/unit/search_pipeline -q`

Expected: all pass and no duplicate retrieval occurs.

- [ ] **Step 5: Commit**

```bash
git add src/internal/mcp_server/tools/chat.py tests/unit/test_mcp_server.py tests/unit/test_context_pipeline.py
git commit -m "fix: ground MCP answers in authorized evidence"
```

### Task 5: Update documentation and run security regression verification

**Files:**
- Modify: `docs/mcp.md`
- Modify: `src/internal/mcp_server/README.md`
- Modify: `tests/integration/tests/mcp/test_mcp_server_search.py`

**Interfaces:**
- Consumes: authenticated web retrieval behavior from Tasks 1–4.
- Produces: documentation and integration coverage matching the implemented trust boundary.

- [ ] **Step 1: Strengthen integration assertions**

Keep the existing cross-user and document-set scenarios. Add an assertion that an invalid token cannot call the indexed-document tool and ensure the blocked user receives no restricted document content from raw search, raw retrieval, or grounded answer tools.

- [ ] **Step 2: Update MCP documentation**

Document the actual flow:

```text
MCP bearer token → FastMCP verification → authenticated web search endpoint
→ server-derived user/group ACL + optional document-set narrowing
→ internal retrieval → MCP result or grounded synthesis
```

Remove claims that the token is merely validated or that MCP calls raw retrieval directly. State that authentication/backend failures never trigger unfiltered fallback.

- [ ] **Step 3: Run focused verification**

Run: `pytest tests/unit/test_mcp_server.py tests/unit/test_mcp_retrieval_client.py tests/unit/servers/query_and_chat/test_search_backend.py tests/unit/test_tool_registry.py tests/unit/test_tool_arg_validation.py tests/unit/servers/web/test_tool_admin_api.py -q`

Expected: all pass.

- [ ] **Step 4: Run static checks**

Run: `ruff check src/internal/mcp_server src/internal/servers/query_and_chat tests/unit/test_mcp_server.py tests/unit/test_mcp_retrieval_client.py tests/unit/servers/query_and_chat/test_search_backend.py`

Run: `ruff format --check src/internal/mcp_server src/internal/servers/query_and_chat tests/unit/test_mcp_server.py tests/unit/test_mcp_retrieval_client.py tests/unit/servers/query_and_chat/test_search_backend.py`

Run: `git diff --check main...HEAD`

Expected: all exit successfully.

- [ ] **Step 5: Commit**

```bash
git add docs/mcp.md src/internal/mcp_server/README.md tests/integration/tests/mcp/test_mcp_server_search.py
git commit -m "docs: explain authenticated MCP retrieval"
```
