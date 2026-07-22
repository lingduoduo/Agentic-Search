# Docs Sync (Direct Surfaces) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sync `search-engine.md`, `chat-engine.md`, `tool-engine.md`, and `README.md` to the merged direct-surfaces / transcript / approval work.

**Architecture:** Documentation only — additive edits in existing sections, verified against merged `main`.

## Global Constraints

- Branch `docs/sync-direct-surfaces` (off `main`).
- Docs only; no code changes.
- Every named endpoint/behavior must exist on `main` (verify before writing).
- Relative Markdown links must resolve.

---

## Task 1: `docs/search-engine.md` — dedicated search surface

- [ ] **Step 1** — After the "Request routing" section, add:

```markdown
## Dedicated search surface (`/search/*`)

Beyond the unified `/api/agent`, search has its own retrieval-only surface,
parallel to `/chat/*` and `/tool/*`:

- `POST /search/send-search-message` — runs the (optionally expanded) retrieval
  pipeline and returns the ranked documents with their executed queries. No LLM
  synthesis. Returns JSON, or a newline-delimited JSON stream when `stream:true`.
- `POST /search/search-flow-classification` — keyword-vs-chat routing hint.
- `GET /search/search-history` — past sessions for the caller.

In the web UI, the **Search** tab drives `send-search-message` and renders the
returned documents directly (no answer panel).
```

- [ ] **Step 2** — Commit: `git add docs/search-engine.md && git commit -m "docs(search): document the dedicated /search/* surface"`

---

## Task 2: `docs/chat-engine.md` — transcript note

- [ ] **Step 1** — In the "Direct chat surface" section, after the line that ends
"...The runner is `src/internal/servers/web/plain_chat_runner.py`.", append a
sentence to the "In the web UI, the **Chat** tab drives this endpoint." line so
it reads:

```markdown
In the web UI, the **Chat** tab drives this endpoint and renders a running
transcript of the session's turns (accumulated client-side as you chat).
```

(If the existing sentence differs, edit in place to add the transcript clause.)

- [ ] **Step 2** — Commit: `git add docs/chat-engine.md && git commit -m "docs(chat): note the Chat tab running transcript"`

---

## Task 3: `docs/tool-engine.md` — transcript + inline approvals

- [ ] **Step 1** — Replace the existing UI sentence ("In the web UI, the **Tool
Agent** tab (Assistant | Tool Agent switcher) drives this endpoint with a live
tool-call trace.") with:

```markdown
In the web UI, the **Tool Agent** tab drives this endpoint and renders a running
transcript: each turn interleaves its live tool-call trace, and gated tools show
an inline approval prompt (Approve / Deny) before they run.
```

- [ ] **Step 2** — Commit: `git add docs/tool-engine.md && git commit -m "docs(tool): note transcript + inline approval prompts"`

---

## Task 4: `README.md` — four surfaces + frontend note

- [ ] **Step 1** — In "## What it provides", replace the UI bullet
("- A React chat UI with streaming responses, source inspection, and observability surfaces")
with:

```markdown
- A React UI with four surfaces — an auto-routing Assistant plus direct Search, Chat, and Tool Agent tabs — with streaming responses, a running conversation transcript, source inspection, and observability panels
```

- [ ] **Step 2** — In "## Search engine", append to the paragraph:
"It also exposes a dedicated retrieval-only surface at `POST /search/send-search-message` (the **Search** tab)."

- [ ] **Step 3** — In "## Chat engine", append:
"A direct `POST /chat/send-chat-message` endpoint (the **Chat** tab) calls the local model with no retrieval, streaming a multi-turn transcript."

- [ ] **Step 4** — In "## Tool engine", append:
"A dedicated `POST /tool/send-tool-message` surface (the **Tool Agent** tab) streams tool calls, gates tools with approval prompts, and fetches the web via a serpapi→browser cascade."

- [ ] **Step 5** — In "### 3. Start the frontend", after "Open <http://127.0.0.1:5173>.", add:
"The header switches between the **Assistant**, **Search**, **Chat**, and **Tool Agent** surfaces."

- [ ] **Step 6** — Commit: `git add README.md && git commit -m "docs(readme): document the four direct surfaces + transcript"`

---

## Task 5: Verify links

- [ ] **Step 1** — Confirm relative links still resolve:
`grep -oE "\]\(([^)]+\.md[^)]*)\)" docs/search-engine.md docs/chat-engine.md docs/tool-engine.md README.md` and eyeball each target exists.
- [ ] **Step 2** — `git diff --check` clean.

---

## Self-Review

- **Spec coverage:** dedicated search surface (Task 1) ✓; chat transcript (Task 2) ✓; tool transcript + approvals (Task 3) ✓; README four surfaces + endpoints + frontend note (Task 4) ✓; links (Task 5) ✓.
- **Placeholder scan:** none.
- **Accuracy:** endpoints (`/search/send-search-message`, `/search/search-history`, `/chat/send-chat-message`, `/tool/send-tool-message`) verified against `main` before writing.
