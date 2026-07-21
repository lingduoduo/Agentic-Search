# Plan: tool-engine doc extraction

**Spec:** ../specs/2026-07-21-tool-engine-doc-design.md

1. Write `docs/tool-engine.md` (capabilities + routing + tool registry + MCP
   relationship + cross-links) → verify: file renders, links resolve.
2. Add a `## Tool engine` pointer section after `## Search engine` in the README →
   verify: `grep -n "tool-engine.md" README.md` shows the pointer.
3. Add `docs/tool-engine.md` to the README Documentation list → verify: entry
   present.
4. Keep `docs/mcp.md` unchanged → verify: no diff to mcp.md.
5. Sanity-check no dangling links → verify: `request-routing.md` and `mcp.md`
   exist on disk.
