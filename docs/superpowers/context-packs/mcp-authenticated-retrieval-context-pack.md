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

…

## Implementation Plan Context

### Task 1: Enforce authenticated filters in the existing web search endpoint

**Files:**
- Modify: `src/internal/servers/query_and_chat/search_backend.py`
- Create: `tests/unit/servers/query_and_chat/test_search_backend.py`

**Interfaces:**
- Consumes: `user_from_headers(headers) -> AuthenticatedUser | None`, `build_user_only_filters(principal_id, email=..., group_ids=...) -> SearchFilters`
- Produces: `_authenticated_search_filters(request: Request, requested: SearchFilters | None) -> SearchFilters`; non-streaming `POST /search/send-search-message` applies server-derived `access_acl` and caller narrowing fields.

- [ ] **Step 1: Write failing endpoint tests**

…

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

…

### Task 3: Migrate raw MCP search and retrieval tools

**Files:**
- Modify: `src/internal/mcp_server/tools/search.py`
- Modify: `src/internal/mcp_server/tools/research.py`
- Test: `tests/unit/test_mcp_server.py`

**Interfaces:**
- Consumes: `authenticated_retrieve(query, top_k=..., document_set_names=...)`
- Produces: unchanged `search_indexed_documents` result shape and unchanged `retrieve_documents` result shape.

- [ ] **Step 1: Replace mocks with failing authenticated-client expectations**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
