# Tool-Approval Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The Tool Agent surface gates tools interactively (approval prompts), at parity with the `/api/agent/stream` auto-router path.

**Architecture:** Wire an `on_approval` into `/tool/send-tool-message`'s streaming queue using the existing `_request_tool_approval` + `ToolApprovalBroker`; the frontend reuses `ToolApprovalCard` + `submitToolApproval` and the existing decision endpoint.

**Tech Stack:** FastAPI, pytest (backend); React 19, TypeScript, Vitest (frontend).

## Global Constraints

- Branch `feat/tool-approval-parity` (off current `main`, which includes #449).
- Reuse the existing decision endpoint `POST /api/agent/approvals/{approval_id}` — do NOT add a new one.
- Approvals apply to the STREAMING path only; the non-streaming path keeps `on_approval=None`.
- Import `_request_tool_approval` function-locally (it lives in `app.py`) — a module-level import reintroduces the circular import.
- Backend tests mount only the router on a bare app; do not boot the full app. Run vitest from `web/`.
- `ruff check . --fix && ruff format .` before backend commits; `npm run typecheck` before frontend commits.

---

## Task 1: Wire `on_approval` into the tool streaming endpoint

**Files:**
- Modify: `src/internal/servers/query_and_chat/tool_backend.py`
- Test: `tests/unit/test_tool_backend.py`

**Interfaces:**
- Consumes: `_request_tool_approval(broker, owner_user_id, approval_request, queue)` from `src.internal.servers.web.app`; `app.state.tool_approval_broker`.

- [ ] **Step 1: Thread `on_approval` through `_run`**

In `tool_backend.py`, change `_run`'s signature and its `_run_tool_agent` call:

```python
        async def _run(on_turn=None, on_approval=None):
            answer, _citations, documents, _intent, extra = await _run_tool_agent(
                body.message,
                manager=manager,
                tokenizer=tokenizer,
                search_url=search_url,
                history=history,
                resolved=resolved,
                on_turn=on_turn,
                on_approval=on_approval,
                with_search_tool=body.run_search_tool,
            )
            answer = answer or extra.pop("_assistant_fallback", "")
            tool_calls = extra.get("tool_calls", [])
            return answer, tool_calls, extra.get("num_turns", 0)
```

(The non-stream branch keeps calling `await _run()` → `on_approval=None`.)

- [ ] **Step 2: Build `on_approval` inside `_gen` and pass it to `_run`**

In `_gen`, after the `queue` is created and `on_turn` is defined, add before the `task = asyncio.create_task(...)` line:

```python
            broker = getattr(http_request.app.state, "tool_approval_broker", None)
            on_approval = None
            if user is not None and not user.is_anonymous and broker is not None:
                from src.internal.servers.web.app import _request_tool_approval

                async def on_approval(approval_request):
                    return await _request_tool_approval(
                        broker, user.id, approval_request, queue
                    )

            task = asyncio.create_task(_run(on_turn=on_turn, on_approval=on_approval))
```

(Replace the existing `task = asyncio.create_task(_run(on_turn=on_turn))` line.)

No other change to `_gen` — the drain loop already forwards `approval_required`
queue items to the client as SSE, and the existing
`POST /api/agent/approvals/{approval_id}` endpoint resolves the broker.

- [ ] **Step 3: Write the failing approval test**

Append to `tests/unit/test_tool_backend.py`:

```python
def test_send_tool_message_emits_approval_required(monkeypatch):
    import asyncio
    from dataclasses import dataclass
    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    @dataclass
    class _View:
        id: str
        tool_name: str
        arguments: dict
        expires_at: str

    class _Broker:
        async def request(self, owner_user_id, approval_request, on_registered=None):
            if on_registered:
                on_registered(_View(id="ap1", tool_name="web_search", arguments={}, expires_at="2030-01-01T00:00:00Z"))
            return "approve"

    captured = {}

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        captured["on_approval"] = on_approval
        if on_approval is not None:
            await on_approval(object())  # triggers the broker → approval_required
        tc = ToolCallView(tool_name="web_search", status="completed", arguments={},
                          result_summary="ok", latency_ms=1, error=None)
        return ("done", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    monkeypatch.setattr(tool_backend, "_run_tool_agent", fake_run_tool_agent)

    app = _make_app(with_model=True)
    app.state.tool_approval_broker = _Broker()
    # Force an authenticated user so on_approval is wired.
    from src.internal.servers.users import api as users_api
    class _User: id = "u1"; is_anonymous = False; email = "u@x"
    monkeypatch.setattr(users_api, "resolve_request_user", lambda *a, **k: _User())

    client = TestClient(app)
    with client.stream("POST", "/tool/send-tool-message", json={"message": "go", "stream": True}) as resp:
        events = [json.loads(l[len("data:"):].strip()) for l in resp.iter_lines() if l.startswith("data:")]

    assert captured["on_approval"] is not None
    assert any(e["type"] == "approval_required" and e["approval"]["id"] == "ap1" for e in events)


def test_no_broker_means_on_approval_none(monkeypatch):
    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    captured = {}

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        captured["on_approval"] = on_approval
        tc = ToolCallView(tool_name="x", status="completed", arguments={}, result_summary="", latency_ms=1, error=None)
        return ("done", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    monkeypatch.setattr(tool_backend, "_run_tool_agent", fake_run_tool_agent)
    app = _make_app(with_model=True)  # no broker set / anonymous user
    app.state.tool_approval_broker = None
    client = TestClient(app)
    with client.stream("POST", "/tool/send-tool-message", json={"message": "go", "stream": True}) as resp:
        list(resp.iter_lines())
    assert captured["on_approval"] is None
```

Note: verify `resolve_request_user`'s import path used by `tool_backend` (it imports `from src.internal.servers.users.api import resolve_request_user`). Patch the name where `tool_backend` looks it up — i.e. `monkeypatch.setattr(tool_backend, "resolve_request_user", ...)` if it's imported into the module namespace; otherwise patch `users_api.resolve_request_user`. Confirm by reading the import in `tool_backend.py` and adjust the monkeypatch target accordingly.

- [ ] **Step 4: Run RED → GREEN**

Run: `pytest tests/unit/test_tool_backend.py -v`
Expected: the two new tests fail first (on_approval always None), then pass after Steps 1–2. Existing tool_backend tests still pass.

- [ ] **Step 5: Import smoke + lint + commit**

```bash
python -c "from src.internal.servers.web.app import create_web_app; create_web_app()"
ruff check . --fix && ruff format .
git add src/internal/servers/query_and_chat/tool_backend.py tests/unit/test_tool_backend.py
git commit -m "feat(tool-agent): gate tools via approval broker on the streaming endpoint"
```

---

## Task 2: Frontend — approval events + card in ToolAgentView

**Files:**
- Modify: `web/src/types.ts` (add `approval_required` to `ToolStreamEvent`)
- Modify: `web/src/components/ToolAgentView.tsx`
- Test: `web/src/components/__tests__/ToolAgentView.test.tsx`

**Interfaces:**
- Consumes: `ToolApprovalView`, `submitToolApproval`, `ToolApprovalCard` (all existing).

- [ ] **Step 1: Add the approval variant to `ToolStreamEvent`**

In `web/src/types.ts`, extend the union (import/reuse `ToolApprovalView`, already defined in this file):

```typescript
export type ToolStreamEvent =
  | { type: "progress"; turn: number; text: string }
  | ({ type: "tool_call" } & ToolCallTraceView)
  | { type: "answer"; text: string }
  | { type: "approval_required"; approval: ToolApprovalView }
  | { type: "done"; session_id: string; tool_calls: ToolCallTraceView[]; num_turns: number }
  | { type: "error"; detail: string };
```

- [ ] **Step 2: Render pending approvals in ToolAgentView**

In `web/src/components/ToolAgentView.tsx`:

1. Add imports:
```typescript
import type { ConversationTurn, ToolApprovalView } from "../types";
import { submitToolApproval } from "../api";
import { ToolApprovalCard } from "./ToolApprovalCard";
```
2. Add state: `const [pendingApprovals, setPendingApprovals] = useState<ToolApprovalView[]>([]);`
3. In `submit()`, reset it alongside the others: `setPendingApprovals([]);` (add near `setError(null)`).
4. In the event loop, handle the new event:
```typescript
        else if (e.type === "approval_required")
          setPendingApprovals((a) => [...a, e.approval]);
```
5. On `done` and `error`, also clear approvals: add `setPendingApprovals([]);` in both branches. Clear it in the `catch` too.
6. Render the cards between the `<Transcript>` and the composer:
```tsx
      {pendingApprovals.map((approval) => (
        <ToolApprovalCard
          key={approval.id}
          approval={approval}
          onDecision={(decision) =>
            submitToolApproval(approval.id, decision).finally(() =>
              setPendingApprovals((a) => a.filter((p) => p.id !== approval.id)),
            )
          }
        />
      ))}
```

- [ ] **Step 3: Add the frontend test**

Append to `web/src/components/__tests__/ToolAgentView.test.tsx`:

```typescript
it("shows an approval card and posts the decision", async () => {
  const submitSpy = vi.spyOn(api, "submitToolApproval").mockResolvedValue({});
  async function* fake() {
    yield {
      type: "approval_required",
      approval: { id: "ap1", tool_name: "web_search", arguments: {}, expires_at: "2030-01-01T00:00:00Z" },
    } as const;
    yield { type: "answer", text: "ok" } as const;
    yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
  }
  vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

  render(<ToolAgentView />);
  fireEvent.change(screen.getByLabelText("Tool agent message"), { target: { value: "go" } });
  fireEvent.click(screen.getByText("Send"));

  const approveBtn = await screen.findByRole("button", { name: /approve/i });
  fireEvent.click(approveBtn);
  await waitFor(() => expect(submitSpy).toHaveBeenCalledWith("ap1", "approve"));
});
```

Note: confirm `ToolApprovalCard`'s approve control is a button with accessible name matching `/approve/i`; if the label differs (e.g. "Allow"), match the real text. Because `done` clears `pendingApprovals`, the card may unmount right after the decision — assert on the `submitSpy` call (already captured), not on the card staying mounted.

- [ ] **Step 4: Typecheck + full suite**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: no type errors; all pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/components/ToolAgentView.tsx web/src/components/__tests__/ToolAgentView.test.tsx
git commit -m "feat(web): Tool Agent renders approval prompts and posts decisions"
```

---

## Task 3: Docs

- [ ] **Step 1** — In `docs/tool-engine.md`, under the `/tool/*` section, add:

```markdown
Gated tools prompt for approval on the streaming path, at parity with the
auto-router: the endpoint emits `approval_required` events through the
`ToolApprovalBroker` and waits for the user's decision via the shared
`POST /api/agent/approvals/{approval_id}` endpoint (authenticated users only).
```

- [ ] **Step 2** — Commit: `git add docs/tool-engine.md && git commit -m "docs: note Tool Agent approval gating"`

---

## Self-Review

- **Spec coverage:** backend `on_approval` wired on the streaming path only (Task 1) ✓; reuse of broker + decision endpoint (Task 1) ✓; anonymous/no-broker → `on_approval=None` (Task 1 test) ✓; frontend `approval_required` variant + card + decision posting + clear-on-end (Task 2) ✓; docs (Task 3) ✓.
- **Placeholder scan:** none; two steps flag a real verify-then-adapt (the `resolve_request_user` monkeypatch target; the approve-button label).
- **Type consistency:** `ToolApprovalView` reused in the union and the view; `submitToolApproval(id, decision)` signature matches api.ts; the `on_approval` kwarg name matches `_run_tool_agent`.
