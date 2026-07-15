# Generated Context Pack

# Mcp Authenticated Retrieval

## Sources

- [Specification: 2026-07-14-mcp-authenticated-retrieval-design.md](../specs/2026-07-14-mcp-authenticated-retrieval-design.md)
- [Plan: 2026-07-14-mcp-authenticated-retrieval.md](../plans/2026-07-14-mcp-authenticated-retrieval.md)

## Specification Context

### Goal

Ensure every MCP tool that reads private indexed documents carries the current authenticated request context into the web backend, where existing user, group, connector, and document-set access controls are enforced.

This change covers `search_indexed_documents`, `retrieve_documents`, and `ask_agentic_search`. Public web search, URL fetching, and query expansion remain outside this scope.

### Architecture

Introduce one internal MCP client boundary responsible for authenticated indexed-document operations. It obtains the current FastMCP `AccessToken` through `require_access_token()`, constructs an `Authorization: Bearer ...` header, and calls the web backend URL from `build_web_base_url()`.

The web backend remains responsible for:

- validating the token on the retrieval request;
- resolving the authenticated user and group membership;
- validating requested document sets against user access;
- building the existing internal retrieval filters;
- invoking retrieval and returning only authorized evidence.

The MCP tool modules translate the authenticated backend response into their existing public MCP result shapes. No MCP tool name or successful result schema changes.

### Testing

Use test-driven development with regressions that first fail against the current direct-retrieval implementation:

- all three indexed-document tools forward the current bearer token;
- `search_indexed_documents` forwards non-empty document-set names;
- an empty document-set list normalizes to no explicit restriction;
- missing authentication prevents any backend or raw retrieval call;
- `401` and `403` do not fall back to raw retrieval;
- authenticated empty results retain the existing successful empty-result contract;
- existing MCP result shapes remain compatible;
- integration coverage proves one user cannot retrieve another user's restricted documents.

Focused MCP, tool-registry, authentication, and document-set integration tests must remain green.

## Implementation Plan Context

### Global Constraints

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
