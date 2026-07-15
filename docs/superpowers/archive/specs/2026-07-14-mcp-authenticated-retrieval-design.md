# Authenticated MCP Retrieval Design

## Goal

Ensure every MCP tool that reads private indexed documents carries the current authenticated request context into the web backend, where existing user, group, connector, and document-set access controls are enforced.

This change covers `search_indexed_documents`, `retrieve_documents`, and `ask_agentic_search`. Public web search, URL fetching, and query expansion remain outside this scope.

## Current problem

The MCP bearer token is validated against the web backend's `/me` endpoint, but the retrieval tools then call the raw retrieval service without authentication or authorization filters. `search_indexed_documents` also accepts `document_set_names` while explicitly ignoring it. As a result, authentication proves that a caller exists but does not scope private retrieval to that caller.

The MCP documentation and integration tests expect the web backend to manage access controls, so direct unfiltered retrieval violates the intended trust boundary.

## Considered approaches

### Authenticated web-backend delegation (selected)

The MCP server forwards the current bearer token and retrieval parameters to an authenticated web endpoint. The web backend resolves the caller and applies its existing ACL filters before querying retrieval.

This adds one internal HTTP hop but keeps authorization in a single authority and prevents MCP from falling back to unfiltered retrieval.

### Build ACL filters inside MCP

The token verifier could copy identity claims into the MCP access token and MCP could reproduce web authorization logic. This avoids the extra hop but duplicates permission rules and creates a second security boundary that can drift.

### Authenticate the raw retrieval service

The token could be forwarded directly to the retrieval server. That would be a smaller call-path change, but the retrieval service does not own user/group/document-set authorization. It would require moving or duplicating web-layer policy.

## Architecture

Introduce one internal MCP client boundary responsible for authenticated indexed-document operations. It obtains the current FastMCP `AccessToken` through `require_access_token()`, constructs an `Authorization: Bearer ...` header, and calls the web backend URL from `build_web_base_url()`.

The web backend remains responsible for:

- validating the token on the retrieval request;
- resolving the authenticated user and group membership;
- validating requested document sets against user access;
- building the existing internal retrieval filters;
- invoking retrieval and returning only authorized evidence.

The MCP tool modules translate the authenticated backend response into their existing public MCP result shapes. No MCP tool name or successful result schema changes.

## Data flow

1. An MCP client calls one of the three indexed-document tools with its bearer token.
2. FastMCP verifies the token through `AgenticSearchTokenVerifier`.
3. The tool reads the verified token from request-local FastMCP context.
4. The shared authenticated retrieval client forwards the token, query, result limit, and optional document-set names to the web backend.
5. The web backend derives ACL filters and queries internal retrieval.
6. The authorized documents return to MCP and are formatted as search results, raw documents, or a grounded answer.

`document_set_names=None` and an empty list both mean no caller-specified document-set restriction; normal user ACLs still apply. A non-empty list narrows retrieval to accessible named sets. It never broadens access.

## Tool behavior

### `search_indexed_documents`

Forward `query`, the fixed result limit, and `document_set_names`. Continue accepting the other compatibility parameters, but do not claim that document-set filtering is ignored. Return the existing `results` shape.

### `retrieve_documents`

Forward `query` and `top_k` through the authenticated client. Return the existing full-document and score shape.

### `ask_agentic_search`

Obtain authorized evidence through the same authenticated boundary before synthesis. LLM generation remains local to the MCP service unless the existing web API already provides the equivalent grounded-answer contract. In either case, generation may use only the authorized evidence returned for this request.

## Failure handling

- Missing request-local access token: fail the tool explicitly; do not call retrieval.
- Web backend `401` or `403`: return an authentication/authorization tool error; do not retry against raw retrieval.
- Backend timeout or `5xx`: return a backend-unavailable tool error.
- Empty authorized result set: return a successful empty result, not an infrastructure error.
- Invalid or inaccessible document-set names: preserve the web backend's safe rejection or empty-result behavior.

No failure path may call the raw retrieval URL without authenticated context.

## Compatibility

- Existing MCP tool names and successful response fields remain stable.
- Web search and URL-fetch tools are unchanged.
- The raw retrieval APIs remain available to their existing internal callers.
- The MCP client still chooses which exposed tool to invoke; this design changes authorization, not tool-selection policy.

## Testing

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
