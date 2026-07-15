# Generated Context Pack

# Streaming UX Design Spec

## Sources

- [Specification: 2026-06-16-streaming-ux-design.md](../specs/2026-06-16-streaming-ux-design.md)

## Specification Context

### 2. Architecture

The agent loop receives an `on_turn` async callback. It calls it after each completed turn. The stream endpoint converts each call into an SSE `progress` event. This is the only coupling point — the rest of the agent loop is unchanged.

---

### 7. Testing Strategy

- **`tests/unit/agents/test_on_turn_callback.py`** — monkeypatch `SearchAgentLoop._step`, assert `on_turn` is called once per turn with correct `(turn, tool_name, doc_count)` args; same for `ToolAgentLoop`
- **`tests/unit/servers/web/test_stream_agent.py`** — `TestClient` + `httpx` streaming: assert `progress` events arrive before `done`; assert existing `/api/agent` still returns a plain JSON blob
- **Frontend:** `web/src/components/__tests__/AnswerPanel.test.tsx` — render with `progressSteps=[{turn:1,text:"search…"}]`, assert log is visible; render with `progressSteps=[]` + `totalTurns=2`, assert collapsed summary shows

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
