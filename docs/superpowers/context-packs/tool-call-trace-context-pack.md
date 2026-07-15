# Generated Context Pack

# Tool Call Trace

## Sources

- [Specification: 2026-06-16-tool-call-trace-design.md](../specs/2026-06-16-tool-call-trace-design.md)
- [Plan: 2026-06-17-tool-call-trace.md](../plans/2026-06-17-tool-call-trace.md)

## Specification Context

### 2. Architecture

`ToolCallView` is a Pydantic model added to `app.py` alongside the existing response models. The parsing logic extends the existing `action_trace` loop in `_run_auto_routed`.

---

### 4.2 New `ToolCallTracePanel` component

**File:** `web/src/components/ToolCallTracePanel.tsx`

## Implementation Plan Context

### Task 1: Backend — `arguments` field + `ToolCallView` model + trace parsing

**Files:**
- Modify: `src/agents/state.py`
- Modify: `src/agents/tool_calling.py`
- Modify: `src/internal/servers/web/app.py`
- Create: `tests/unit/servers/web/test_tool_trace.py`

### Task 2: Frontend types + `ToolCallTracePanel` component

**Files:**
- Modify: `web/src/types.ts`
- Create: `web/src/components/ToolCallTracePanel.tsx`
- Create: `web/src/components/__tests__/ToolCallTracePanel.test.tsx`

- [x] **Step 1: Write failing component tests**

Create `web/src/components/__tests__/ToolCallTracePanel.test.tsx`:

- [x] **Step 2: Run failing tests**

Expected: tests FAIL (module not found / no export).

- [x] **Step 3: Add `ToolCallTraceView` to `web/src/types.ts`**

Append to the end of `web/src/types.ts`:

Also add `tool_calls?` to `AgentExperienceResponse`:
And add `tool_calls?` to `SSEDoneEvent`:
- [x] **Step 4: Create `web/src/components/ToolCallTracePanel.tsx`**

- [x] **Step 5: Run component tests**

…

### Task 3: Wire `App.tsx` + SSE streaming + CSS

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`

- [x] **Step 1: Read `web/src/App.tsx`** to understand current structure before editing.

- [x] **Step 2: Update `App.tsx`**

Add import at the top:
Add state in the component:
In `handleNewSession` (the function that resets state), add:
In the `streamAgent` loop, find the `} else if (event.type === "done") {` block and add:
In the JSX, inside `.results-layout`, after `<AnswerPanel ...>` and before the Sources section, add:
- [x] **Step 3: Append CSS to `web/src/styles.css`**

Append at the end:

- [x] **Step 4: Run full frontend test suite**

Expected: all tests pass (65+).

- [x] **Step 5: Run typecheck**

…

### Task 4: Also forward `tool_calls` in the SSE streaming path

**Files:**
- Modify: `src/internal/servers/web/app.py`

The `stream_agent` SSE endpoint emits a `done` event from `_run_agent_impl` result. Currently the `done` event doesn't include `tool_calls`. Fix:

- [x] **Step 1: Find the SSE `done` yield in `stream_agent`**

In `app.py`, find `stream_agent` and the line:
Add `"tool_calls": [tc.model_dump() for tc in result.tool_calls]`:
- [x] **Step 2: Run the full Python unit suite**

Expected: 1815+ passed.

- [x] **Step 3: Commit**

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
