# Spec: Document registry inspection endpoints in tool-engine.md

## Problem
`docs/tool-engine.md` describes the `ToolRegistry`, discovery, and MCP bridge but
omits the read-only inspection surface shipped in PR #445 (`GET /api/debug/tools`,
`POST /api/debug/tools/discover`, backing the Dev Console **Tools** panel). An
accuracy sync of the doc against `src/tools/` and `src/agents/tool/` found every
other claim correct; this was the only gap.

## Scope
Add a short "Inspecting the registry" section after "Tool registry and discovery"
covering the two debug endpoints and what each returns. Documentation only — no
code change.

## Success criteria
- Section names both endpoints with their return shape (`registered` + `catalog`
  for GET; TF-IDF per-stage routing details for discover).
- Matches the actual handlers in `src/internal/servers/web/debug_router.py`.
- No other doc content changed (surgical).
