# Generated Context Pack

# Intent Routed Search Chat

## Sources

- [Specification: 2026-06-15-intent-routed-search-chat-design.md](../specs/2026-06-15-intent-routed-search-chat-design.md)
- [Plan: 2026-06-15-intent-routed-search-chat.md](../plans/2026-06-15-intent-routed-search-chat.md)

## Specification Context

### Goal

- Single input box — no mode selector visible to users.
- The system routes automatically: the `ToolAgentLoop` (local model) acts as the universal router when available, expressing intent by which tool it calls.
- Clean fallback when no local model is configured.
- Errors are specific and actionable, never "Agent search failed".

---

### Out of Scope

- Training the `IntentPipeline` model with search/chat/tool labels — it stays as a standalone training artifact.
- Streaming progress events for `search_tool` / `rag_tool` calls inside `ToolAgentLoop` — existing SSE stream is sufficient.
- Changing the session or history model.

---

### `SearchComposer.test.tsx` — update existing

The existing test `"shows all six mode options"` and the `mode` prop must be removed/updated since the mode dropdown is gone.

| Test | What it asserts |
|---|---|
| `"no mode dropdown is rendered"` | `screen.queryByLabelText(/entry point/i)` is `null` |
| `"renders retrieval URL and topK fields"` | Both inputs are present without a mode selector |
| `"submit enabled when query has content"` | Unchanged from existing test |
| `"submit disabled when loading"` | Unchanged from existing test |
| `"Cmd+Enter submits the form"` | `userEvent.keyboard('{Meta>}{Enter}{/Meta}')` triggers `onSubmit` |

…

## Implementation Plan Context

### Task 1: Rule-based classifier + trajectory intent inference

**Files:**
- Create: `src/internal/servers/web/intent_routing.py`
- Create: `tests/unit/test_intent_routing.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist)**

Expected: `ModuleNotFoundError: No module named 'src.internal.servers.web.intent_routing'`

- [ ] **Step 3: Create `intent_routing.py`**

- [ ] **Step 4: Run tests — expect PASS**

Expected: 12 tests pass.

- [ ] **Step 5: Commit**

---

### Task 2: Routing tool builders

**Files:**
- Create: `src/tools/routing_tools.py`
- Modify: `src/tools/__init__.py`
- Test: `tests/unit/test_intent_routing.py` (append)

- [ ] **Step 1: Write failing tests (append to test file)**

- [ ] **Step 2: Run tests — expect ImportError**

Expected: `ModuleNotFoundError: No module named 'src.tools.routing_tools'`

- [ ] **Step 3: Create `routing_tools.py`**

- [ ] **Step 4: Export from `src/tools/__init__.py`**

Add these two lines at the end of `src/tools/__init__.py`:

- [ ] **Step 5: Run tests — expect PASS**

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

---

### Task 3: Update request/response models

**Files:**
- Modify: `src/internal/servers/web/app.py` (models only — lines 148–176)
- Test: `tests/unit/servers/web/test_web_experience_app.py` (append)

- [ ] **Step 1: Write failing test (append to existing test file)**

- [ ] **Step 2: Run test — expect KeyError (field missing)**

Expected: FAIL — `"intent" not in data`

- [ ] **Step 3: Update models in `app.py`**

Change `AgentExperienceRequest.mode` (around line 161):
Add `intent` to `AgentExperienceResponse` (around line 171, after `hook_metadata`):
- [ ] **Step 4: Run test — expect PASS**

Expected: PASS.

- [ ] **Step 5: Run full existing test suite to catch regressions**

…

### Final verification

- [ ] **Run complete test suite**

```bash
pytest tests/unit/ -v --tb=short
cd web && npm run test && npm run typecheck
```

Expected: all pass.

- [ ] **Smoke test the stack**

```bash
# Terminal 1
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3
cd web && npm run dev
```

Open `http://127.0.0.1:5173`. Verify:
- No mode dropdown visible
- Typing "explain FAISS" → answer panel prominent, intent badge says "Answered · N citations"
- Typing "procurement process" → source grid prominent, intent badge says "Searched · N sources"

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
