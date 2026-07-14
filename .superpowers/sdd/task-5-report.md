# Task 5 report: authenticated MCP retrieval documentation and regression coverage

## Changes

- Documented the actual trust boundary in `docs/mcp.md` and the MCP server README:
  FastMCP verifies the bearer credential, indexed-document tools forward it only to
  the authenticated web search endpoint, and the web backend derives user/group ACLs
  before internal retrieval.
- Documented fail-closed behavior: authentication, authorization, transport, and
  backend errors never trigger unfiltered raw retrieval.
- Described grounding accurately as citation-label and answer/evidence-overlap
  validation, not a hard no-hallucination guarantee.
- Repaired the existing integration flow's undefined payload/results variables.
- Strengthened the Enterprise ACL scenario so a blocked user receives no restricted
  content from `search_indexed_documents`, `retrieve_documents`, or
  `ask_agentic_search`.
- Added a direct streamable-HTTP `tools/call` regression proving an invalid bearer
  token cannot dispatch `search_indexed_documents`.
- Added AST-level assertions that the MCP search, research, and chat modules do not
  import raw retrieval symbols.

## Verification

- Focused tests: `92 passed`
  - `pytest tests/unit/test_mcp_server.py tests/unit/test_mcp_retrieval_client.py tests/unit/servers/query_and_chat/test_search_backend.py tests/unit/test_tool_registry.py tests/unit/test_tool_arg_validation.py tests/unit/servers/web/test_tool_admin_api.py -q`
- MCP server unit tests: `32 passed`
- Integration collection: `4 tests collected`
  - `pytest tests/integration/tests/mcp/test_mcp_server_search.py --collect-only -q`
- Ruff check: passed.
- Ruff format check: 22 files already formatted.
- `git diff --check main...HEAD`: passed.
- `git diff --check`: passed for the uncommitted Task 5 changes.
- Local documentation links checked: `README.md` and `docs/request-routing.md` exist.

## Concerns

- The live MCP integration tests were not executed because they require the external
  MCP server, web API, database/indexing services, and Enterprise ACL configuration.
  The test module imports and collects successfully.
- Pre-existing modifications to `.superpowers/sdd/task-3-report.md` and
  `.superpowers/sdd/task-4-report.md` were left untouched and are not part of the
  Task 5 commit.
