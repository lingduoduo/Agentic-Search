# Design: visual polish for the four surfaces

Date: 2026-07-21
Status: Approved (brainstorming)
Deliverable 3 of 3 (follow-ups to PR #448)

## Problem

The four-tab experience (Assistant | Search | Chat | Tool Agent) and the new
Chat/Search/Tool views + running transcript ship with **no CSS** — the classes
`surface-switcher`, `transcript`, `turn`, `turn--user`, `turn--assistant`,
`chat-view`, `search-view`, `tool-agent-view` (and their `__composer` children)
have zero rules in `web/src/styles.css`, so they render with browser defaults.

## Goal

A deliberate visual pass so the new surfaces feel native to the app: a proper
tab bar with an active state, chat-style transcript bubbles, and composer rows
consistent with the existing `.composer`. Reuse the existing design tokens.

Decisions locked during brainstorming: **fuller polish** (active-tab states,
transcript bubbles, spacing, responsive), not just minimal consistency.

## Non-goals (YAGNI)

- No new markup / className changes to components — CSS only, so it cannot
  conflict with the in-flight approval-parity PR (#450). Every class already
  exists in the rendered DOM.
- No design-token changes (`:root` stays as-is).
- No changes to already-styled elements (`.composer`, `.icon-button`,
  `.tool-approval-*`, `.error-banner`).

## Design

All rules use the existing tokens: `--bg --panel --panel-soft --ink --muted
--line --accent --accent-strong --shadow --rose`.

- **Surface switcher** (`.surface-switcher`): a segmented tab bar — a flex row,
  gap, wrapping on narrow widths. The buttons already carry `.icon-button` +
  `.active`; add an `.icon-button.active` rule (accent background,
  contrasting text) so the current surface reads as selected.
- **Transcript** (`.transcript`): a vertical stack (`display:flex; flex-flow:
  column; gap`) with a max-width and comfortable line-height.
  - `.turn` — base bubble spacing.
  - `.turn--user` — right-aligned, accent-tinted bubble, self-end.
  - `.turn--assistant` — left-aligned, panel-surface bubble, self-start.
  - `.turn__content` — padding, border-radius, wrap long content.
  - `.turn__progress` — small muted italic list (ephemeral streaming lines).
- **Views** (`.chat-view`, `.search-view`, `.tool-agent-view`): flex column with
  gap; their `__composer` is a row (`input` grows, button fixed) mirroring
  `.composer` padding/border/shadow. Keep `*__answer` styled (legacy safety).
- **Responsive**: transcript and composers use relative widths / `max-width:
  100%`; the switcher wraps; bubbles cap at ~85% width. No horizontal overflow.

## Testing / verification

CSS is not unit-tested. Verification:
- `npm run typecheck` stays clean (no TS impact) and `npm run build` succeeds
  (Vite processes the stylesheet).
- The full frontend test suite stays green (no behavior change).
- Manual visual check when running the stack (documented, not automated).

## Files touched

Modified:
- `web/src/styles.css` — append a "Direct surfaces" section with the rules above.
