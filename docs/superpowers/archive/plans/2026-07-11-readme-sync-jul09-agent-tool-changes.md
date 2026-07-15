# README Sync — Jul-09 Agent/Tool Changes — Plan

Spec: [2026-07-11-readme-sync-jul09-agent-tool-changes-design.md](../specs/2026-07-11-readme-sync-jul09-agent-tool-changes-design.md)

## Steps

1. **Agent Loops bullets** (README `## Features` → "Agent Loops")
   → verify: bullets exist for #395 (tool-error feedback), #397 (arg validation),
   #392 (full-page cap); wording matches merged behavior.

2. **System-preserving crop** (README "Agent Framework & Control Flow", `run()`
   control-flow paragraph) → verify: one sentence notes the crop keeps the system
   prompt when over `prompt_length`.

3. **Conversation-aware search_agent** (README "Web Backend API" dispatch table /
   surrounding prose) → verify: note that `search_agent` mode threads session
   history; the web-reachability known-gap block is unchanged.

4. **Tool category flags** (README `## Features` → "Tool Use") → verify: bullet for
   `Tool.citeable` / `Tool.stopping`.

5. **Lint + anchor check** → verify: `ruff check .` clean; no new broken
   `#`-anchors; `git diff` touches only README + this spec/plan.

6. **Commit, push, PR** → verify: branch pushed, PR opened with a specific title.
