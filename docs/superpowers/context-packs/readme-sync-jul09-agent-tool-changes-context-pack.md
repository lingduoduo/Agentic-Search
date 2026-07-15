# Generated Context Pack

# Readme Sync Jul09 Agent Tool Changes

## Sources

- [Specification: 2026-07-11-readme-sync-jul09-agent-tool-changes-design.md](../archive/specs/2026-07-11-readme-sync-jul09-agent-tool-changes-design.md)
- [Plan: 2026-07-11-readme-sync-jul09-agent-tool-changes.md](../archive/plans/2026-07-11-readme-sync-jul09-agent-tool-changes.md)

## Specification Context

### Scope — which PRs get documented

Document only what changes observable behavior or capability:

| PR | Change | README-worthy? |
|----|--------|----------------|
| #391 | `search_agent` web mode threads session history | Yes — fixes the known single-turn gap |
| #394 | token crop preserves the system prompt on long chats | Yes — correctness behavior |
| #395 | `ToolAgentLoop` feeds tool errors back instead of aborting | Yes — behavior change |
| #397 | `ToolAgentLoop` validates tool arguments before executing | Yes — safety behavior |
| #392 | `max_full_page_chars` caps `<fetch>` full-page observations | Brief — new config knob |
| #399 | `Tool.citeable` / `Tool.stopping` category flags | Brief — new metadata |

…

## Implementation Plan Context

### Overview

Spec: 2026-07-11-readme-sync-jul09-agent-tool-changes-design.md

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
