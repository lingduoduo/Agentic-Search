# README Sync — Jul-09 Agent/Tool Changes — Design

Date: 2026-07-11
Status: Approved
Branch/PR: docs/readme-sync-jul09-agent-tool-changes

## Problem

The README was last synced to the codebase at PR #390 (`a4fae46`, GRPO training
demo + reward-dimensions). Nine PRs landed after it — the 2026-07-09 → 07-11
agent/tool batch (#391–#399) — and none of their concepts appear in the README
(verified: greps for `multiturn`, `session history`, `full-page`, `prompt_length`,
`tool error`, `validate_arguments`, `citeable`, `stopping` all return 0 hits).

Goal: bring the README current for the user-facing and behavioral changes in that
batch, without re-documenting the 28 earlier July PRs that are already reflected.

## Scope — which PRs get documented

Document only what changes observable behavior or capability:

| PR | Change | README-worthy? |
|----|--------|----------------|
| #391 | `search_agent` web mode threads session history | Yes — fixes the known single-turn gap |
| #394 | token crop preserves the system prompt on long chats | Yes — correctness behavior |
| #395 | `ToolAgentLoop` feeds tool errors back instead of aborting | Yes — behavior change |
| #397 | `ToolAgentLoop` validates tool arguments before executing | Yes — safety behavior |
| #392 | `max_full_page_chars` caps `<fetch>` full-page observations | Brief — new config knob |
| #399 | `Tool.citeable` / `Tool.stopping` category flags | Brief — new metadata |
| #393 | centralize citation-label helpers (byte-identical refactor) | No — internal only |
| #396 | delete dead `AgentState` fields | No — internal only |
| #398 | route `ToolAgentLoop` execution through a per-loop `ToolRegistry` | No — internal refactor |

## Non-goals

- No re-documentation of already-synced pre-#390 July work.
- No structural rewrite of the README — surgical edits into existing sections.
- No documentation of pure-internal refactors (#393/#396/#398) that a README
  reader cannot observe.

## Approach

Edit the existing sections rather than adding new ones:

- **Agent Loops** (`## Features` → "Agent Loops") — add bullets for the
  `ToolAgentLoop` error-feedback (#395) and argument-validation (#397) behavior,
  the `SearchAgentLoop` full-page cap (#392), and note system-preserving crop
  (#394) on the shared `AgentLoopBase` path.
- **Agent Framework & Control Flow** — the `run()` control-flow prose already
  describes the shared prompt path; note that the crop is system-preserving (#394).
- **Web Backend API / dispatch** — note `search_agent` mode is now
  conversation-aware (#391); the streamed-search "known gap" section is about
  web-vs-internal reachability, not history, so it is left untouched.
- **Tool Use** (`## Features` → "Tool Use") — add the `citeable`/`stopping` flags
  (#399).

## Success criteria

- Each of #391, #392, #394, #395, #397, #399 is reflected in the correct existing
  README section, accurately (matching the merged behavior, not the spec's
  proposal).
- No claims added for #393/#396/#398.
- `ruff` clean (README-only change, but run the repo checks); no broken internal
  markdown anchors introduced.
