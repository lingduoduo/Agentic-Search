# Surface CSS Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Style the four-tab switcher, the running transcript, and the Chat/Search/Tool views so they feel native, using the existing design tokens.

**Architecture:** CSS-only — append one section to `web/src/styles.css`. No markup changes, so it cannot conflict with the in-flight approval PR (#450).

**Tech Stack:** CSS, Vite.

## Global Constraints

- Branch `feat/surface-css-polish` (off `main`, which has #449's transcript classes).
- CSS-only: touch `web/src/styles.css` only. No `.tsx` edits.
- Reuse tokens `--bg --panel --panel-soft --ink --muted --line --accent --accent-strong --shadow --rose`; do not add or change tokens.
- Do not restyle already-styled elements (`.composer`, `.icon-button` base, `.tool-approval-*`, `.error-banner`); only ADD `.icon-button.active` and the new-class rules.
- `npm run typecheck`, `npm run build`, and `npx vitest run` must all stay green.

---

## Task 1: Append the "Direct surfaces" stylesheet section

**Files:**
- Modify: `web/src/styles.css`

- [ ] **Step 1: Append the CSS**

Append to the end of `web/src/styles.css`:

```css

/* --------------------------------------------------------------------------
   Direct surfaces: tab switcher, transcript bubbles, and the
   Chat / Search / Tool Agent views. (Classes rendered by App.tsx +
   Transcript/ChatView/SearchView/ToolAgentView.)
   -------------------------------------------------------------------------- */

.surface-switcher {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.icon-button.active {
  background: var(--accent);
  border-color: var(--accent-strong);
  color: #fff;
}

.transcript {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
}

.turn {
  display: flex;
  flex-direction: column;
  max-width: 85%;
}

.turn--user {
  align-self: flex-end;
  align-items: flex-end;
}

.turn--assistant {
  align-self: flex-start;
  align-items: flex-start;
}

.turn__content {
  padding: 10px 14px;
  border-radius: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.turn--user .turn__content {
  background: var(--accent);
  color: #fff;
  border-bottom-right-radius: 4px;
}

.turn--assistant .turn__content {
  background: var(--panel);
  color: var(--ink);
  border: 1px solid var(--line);
  border-bottom-left-radius: 4px;
}

.turn__progress {
  margin: 0 0 6px;
  padding-left: 16px;
  color: var(--muted);
  font-size: 0.82rem;
  font-style: italic;
  list-style: disc;
}

.chat-view,
.search-view,
.tool-agent-view {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chat-view__composer,
.search-view__composer,
.tool-agent-view__composer {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: var(--shadow);
}

.chat-view__composer input,
.search-view__composer input,
.tool-agent-view__composer input {
  flex: 1;
  min-height: 42px;
  padding: 0 12px;
}

.chat-view__composer button,
.search-view__composer button,
.tool-agent-view__composer button {
  min-height: 42px;
  padding: 0 18px;
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: #fff;
  font-weight: 750;
  cursor: pointer;
}

.chat-view__composer button:disabled,
.search-view__composer button:disabled,
.tool-agent-view__composer button:disabled {
  opacity: 0.6;
  cursor: default;
}

@media (max-width: 640px) {
  .turn {
    max-width: 100%;
  }
}
```

- [ ] **Step 2: Verify build + typecheck + tests**

Run: `cd web && npm run typecheck && npx vitest run && npm run build`
Expected: typecheck clean; all frontend tests pass; Vite build succeeds (stylesheet compiles, no CSS syntax error).

- [ ] **Step 3: Commit**

```bash
git add web/src/styles.css
git commit -m "feat(web): polish CSS for surface switcher, transcript, and views"
```

---

## Self-Review

- **Spec coverage:** switcher tab bar + active state (`.surface-switcher`, `.icon-button.active`) ✓; transcript bubbles user/assistant + progress (`.transcript`, `.turn*`) ✓; view composers consistent with `.composer` ✓; responsive (max-width, wrap, mobile bubble width) ✓; tokens reused, no token/markup changes ✓.
- **Placeholder scan:** none — literal CSS.
- **Consistency:** every selector targets a class confirmed present in the rendered DOM (`surface-switcher`, `transcript`, `turn--user/assistant`, `turn__content`, `turn__progress`, `{chat,search,tool-agent}-view[__composer]`); `.icon-button.active` matches the switcher buttons' `active` class.
