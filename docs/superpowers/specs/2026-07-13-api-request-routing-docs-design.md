# API Request Routing Documentation Design

## Goal

Make the maintained documentation describe API request routing accurately and consistently, including the search-provider precedence introduced in PR #410.

## Scope

Create `docs/request-routing.md` as the canonical routing contract. Update `README.md` and every maintained top-level guide under `docs/` wherever routing affects that guide's subject. Do not rewrite historical files under `docs/superpowers/specs/` or `docs/superpowers/plans/`; this design document is the only new file in that historical area.

## Canonical routing contract

The new guide will document both `POST /api/agent` and `POST /api/agent/stream` and distinguish two request paths:

1. An explicit `mode` selects its named loop or pipeline.
2. An omitted `mode` invokes the three-way `chat`, `search`, or `tool` router.

For an auto-routed `search` request with `source_provider=auto`, provider precedence is:

1. Internal retrieval.
2. The direct sufficiency gate accepts strong internal evidence immediately.
3. Weak or empty internal evidence falls through to SerpAPI.
4. Empty or unavailable SerpAPI falls through to the configured browser-search service.
5. If no provider returns evidence, the API returns a deterministic no-results or sources-unreachable response with no citations or documents.

The local model must not replace absent search evidence with an internal-knowledge answer on this auto-routed path. Explicit modes retain their documented behavior.

The contract will also explain:

- router decision precedence and representative query classes;
- request fields including `mode`, `source_provider`, `top_k`, filters, session history, and service URL overrides;
- explicit modes and their model or service requirements;
- source-provider semantics, filter constraints, and provider availability;
- response fields including `intent`, citations, documents, route metadata, degradation metadata, and search-mode metadata;
- SSE event ordering and terminal error behavior;
- the distinction between request routing, source-provider selection, and retrieval-backend routing;
- concrete `RAG` and `GRPO` examples showing that corpus coverage affects which provider supplies evidence, not whether an LLM may invent a search answer.

## Documentation structure

- `docs/request-routing.md`: complete and authoritative behavioral reference.
- `README.md`: short routing summary and documentation link.
- `docs/api-reference.md`: API payloads, tables, response metadata, examples, and a concise dispatch summary linked to the canonical guide.
- `docs/architecture.md`: component boundaries and end-to-end request-flow diagram.
- `docs/configuration.md`: environment variables that enable models and search providers, with precedence notes.
- `docs/retrieval.md`: internal, SerpAPI, and browser roles; sufficiency and failure behavior.
- `docs/frontend.md`: how `intent`, streaming events, source cards, and inspector metadata reflect routing.
- `docs/testing.md`: focused routing and fallback test commands.
- `docs/training-and-evaluation.md`: serving inference versus training clarification and explicit-mode implications.
- `docs/mcp.md`: clarify that MCP tool invocation is separate from the web API auto-router, with a link to the canonical guide.

## Consistency rules

- Do not claim that `source_provider=auto` fans out internal and web providers in parallel for auto-routed search.
- Do not describe web retrieval as reachable only through degradation or explicit hybrid mode.
- Do not imply that a local policy model is trained during an API request.
- Use `chat`, `search`, and `tool` for auto-router strategies; reserve `search_agent`, `tool_agent`, and other names for explicit `mode` values.
- Describe browser search as an HTTP browser-search service backed by `playwright-cli`, not as direct Playwright execution inside the web request handler.
- Keep the README concise; detailed behavior belongs in `docs/request-routing.md`.

## Verification

- Search all maintained documentation for stale routing statements and reconcile each hit.
- Validate relative Markdown links.
- Run the routing and web fallback unit suites because the docs include executable examples and exact response semantics.
- Run formatting or documentation checks available in the repository and `git diff --check`.
