# Tool-Agent Human Approval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require the initiating authenticated user to approve every side-effecting or unclassified `ToolAgentLoop` invocation before it executes in a live SSE request.

**Architecture:** Add fail-closed effect metadata to `Tool`, preflight generated calls inside the existing in-memory `ToolAgentLoop`, and inject an asynchronous approval callback. The web app implements that callback with a process-local future broker and an `approval_required` SSE event; the React client posts the decision while the original stream stays open.

**Tech Stack:** Python 3.11+, asyncio, FastAPI/Pydantic, React 19, TypeScript, Vitest, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-06-27-tool-agent-human-approval-design.md`

## Global Constraints

- Approval is per invocation and is never remembered.
- `READ_ONLY` tools execute unchanged; `SIDE_EFFECTING` and `UNSPECIFIED` tools require approval.
- Missing callbacks, anonymous callers, denial, timeout, cancellation, and broker errors fail closed.
- Phase 1 is process-local. Do not add database tables, checkpoints, reconnectable streams, or restart-safe claims.
- Only the initiating authenticated user may decide.
- Browser events and logs receive sanitized summaries, never unrestricted arguments.
- `SearchAgentLoop`, scheduled tasks, direct registry REST invocation, and MCP invocation remain unchanged.

---

### Task 1: Add fail-closed tool effect metadata

**Files:**
- Modify: `src/tools/base.py`
- Modify: `src/tools/api.py`
- Modify: `src/tools/registry.py`
- Modify: `src/tools/routing_tools.py`
- Modify: `src/tools/search.py`
- Modify: `src/tools/__init__.py`
- Test: `tests/unit/test_api_tools.py`
- Test: `tests/unit/test_intent_routing.py`
- Test: `tests/unit/test_search_tools.py`
- Test: `tests/unit/test_tool_registry.py`

**Interfaces:**
- Produces: `ToolEffect(str, Enum)` with `READ_ONLY`, `SIDE_EFFECTING`, and `UNSPECIFIED`.
- Produces: `Tool.effect -> ToolEffect`.
- Produces: effect parameters on `FunctionTool`, `FunctionTool.from_fn`, and `ToolRegistry.tool`.
- Preserves: `ToolSchema.to_dict()`; effect is not shown to the model.

- [ ] **Step 1: Write failing function-tool tests**

Add:

```python
def test_function_tool_defaults_to_unspecified_effect() -> None:
    tool = FunctionTool(lambda: "ok", name="unknown")
    assert tool.effect is ToolEffect.UNSPECIFIED
    assert "effect" not in tool.schema.to_dict()["function"]


def test_registry_decorator_accepts_read_only_effect() -> None:
    registry = ToolRegistry()

    @registry.tool(effect=ToolEffect.READ_ONLY)
    def lookup() -> str:
        return "ok"

    assert registry.get("lookup").effect is ToolEffect.READ_ONLY
```

- [ ] **Step 2: Write failing OpenAPI inference tests**

Build operations for `get`, `head`, `options`, `post`, `put`, `patch`, and
`delete`, then assert read methods are `READ_ONLY` and mutation methods are
`SIDE_EFFECTING`.

- [ ] **Step 3: Verify RED**

Run: `pytest tests/unit/test_tool_registry.py tests/unit/test_api_tools.py -v`

Expected: collection fails because `ToolEffect` does not exist.

- [ ] **Step 4: Implement the effect contract**

In `src/tools/base.py`:

```python
class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    UNSPECIFIED = "unspecified"


class Tool(ABC):
    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.UNSPECIFIED
```

Store and expose the constructor value on `FunctionTool`; thread it through
`from_fn()` and `ToolRegistry.tool()`. Export `ToolEffect` from `src/tools`.

In `ApiRequestTool.effect`, return read-only for `get`, `head`, and `options`,
otherwise side-effecting. Add `head` and `options` to `HTTP_METHODS`.

- [ ] **Step 5: Mark repository-owned search and RAG tools read-only**

Pass `effect=ToolEffect.READ_ONLY` from `build_search_tool`,
`build_search_routing_tool`, and `build_rag_routing_tool`. Add assertions to the
existing builder tests.

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest tests/unit/test_tool_registry.py tests/unit/test_api_tools.py tests/unit/test_intent_routing.py tests/unit/test_search_tools.py -v
git add src/tools tests/unit/test_tool_registry.py tests/unit/test_api_tools.py tests/unit/test_intent_routing.py tests/unit/test_search_tools.py
git commit -m "feat: classify tool side effects"
```

Expected: PASS, then one focused commit.

---

### Task 2: Preflight approvals inside `ToolAgentLoop`

**Files:**
- Modify: `src/agents/tool_calling.py`
- Modify: `src/agents/__init__.py`
- Create: `tests/unit/test_tool_approval.py`

**Interfaces:**
- Consumes: `Tool.effect` from Task 1.
- Produces: `ApprovalDecision`, `ToolApprovalRequest`, and `ToolApprovalCallback`.
- Produces: `ToolAgentLoop.run(..., on_approval: ToolApprovalCallback | None = None)`.
- Produces: skipped results for denied or expired calls.

- [ ] **Step 1: Write failing model and fail-closed tests**

Create `tests/unit/test_tool_approval.py` with the existing minimal tokenizer and
manager pattern. Assert enum values, then run a side-effecting counting tool
without a callback:

```python
assert executions == []
assert output.tool_results[0].status is TaskStatus.SKIPPED
assert output.tool_results[0].error_code == "approval_denied"
```

If `AgentLoopOutput` does not expose `tool_results`, inspect its existing
newline-delimited `action_trace` exactly as web tool-trace tests do; do not add a
second output representation solely for this test.

- [ ] **Step 2: Write failing approve, deny, expiry, and callback-error tests**

Callbacks return each decision or raise. Assert only approval calls
`Tool.execute`; all other cases return `SKIPPED`. Assert requests have unique
opaque IDs, exact arguments, tool names, and timezone-aware expiration.

- [ ] **Step 3: Write failing parallel and continuation tests**

Generate one read-only and two side-effecting calls. Hold both decisions with
events and prove no tool executes until both resolve. Approve one and deny one;
assert the read-only and approved tools execute once. Make the parser produce a
final answer next and prove skipped results are fed back to the model. Separately
prove a genuine failed tool preserves current stop behavior.

- [ ] **Step 4: Verify RED**

Run: `pytest tests/unit/test_tool_approval.py -v`

Expected: failure because approval contracts and callback are absent.

- [ ] **Step 5: Implement contracts and preflight**

Add:

```python
class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    created_at: datetime
    expires_at: datetime


ToolApprovalCallback = Callable[[ToolApprovalRequest], Awaitable[ApprovalDecision]]
```

Add `approval_timeout_seconds: float = 60.0` to `ToolAgentLoopConfig`. Implement
`_request_approval`: read-only returns approved; missing/raising callbacks deny;
requests use `uuid4().hex` and UTC timestamps.

Preflight the truncated call batch with `asyncio.gather`; only then execute
approved/read-only calls. Denied and expired calls become `TaskStatus.SKIPPED`
with stable codes. Change the stop check to stop on `FAILED`, not `SKIPPED`, and
inject safe skipped tool messages for the next model turn.

- [ ] **Step 6: Add additive metrics**

Increment:

```text
tool_approvals_requested
tool_approvals_approved
tool_approvals_denied
tool_approvals_expired
tool_approvals_cancelled
tool_approval_errors
```

Assert decision counters in completed outputs. Since a cancelled loop returns no
output, record cancellation in the broker statistics from Task 3.

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest tests/unit/test_tool_approval.py tests/unit/test_on_turn_callback.py tests/unit/test_state_models.py tests/unit/test_agent_loop.py -v
git add src/agents/tool_calling.py src/agents/__init__.py tests/unit/test_tool_approval.py
git commit -m "feat: gate side-effecting tool calls"
```

Expected: PASS.

---

### Task 3: Build the process-local approval broker

**Files:**
- Create: `src/internal/servers/web/tool_approval.py`
- Create: `tests/unit/servers/web/test_tool_approval_broker.py`

**Interfaces:**
- Consumes: Task 2 approval models.
- Produces: `ToolApprovalBroker.request`, `decide`, and `pending_count`.
- Produces: counters for requested, approved, denied, expired, cancelled, and
  errors.
- Produces: `ApprovalNotFound`, `ApprovalForbidden`, `ApprovalConflict`, and `ApprovalExpired`.
- Produces: `sanitize_tool_arguments(arguments) -> dict[str, object]`.

- [ ] **Step 1: Write failing sanitizer tests**

Assert strings cap at 200 characters plus ellipsis, collections cap at 10,
recursion caps at depth 2, and case-insensitive secret keys are removed:
`password`, `secret`, `token`, `cookie`, `authorization`, `headers`, `api_key`.

- [ ] **Step 2: Write failing lifecycle tests**

Start `broker.request()` as a task and cover approve, deny, wrong user, duplicate
decision, unknown ID, expiry, and cancellation. Assert `pending_count == 0`
after every terminal path and assert the matching counter increments once.

- [ ] **Step 3: Verify RED**

Run: `pytest tests/unit/servers/web/test_tool_approval_broker.py -v`

Expected: collection failure because the module is absent.

- [ ] **Step 4: Implement the broker**

Use:

```python
@dataclass(frozen=True, slots=True)
class ToolApprovalView:
    id: str
    tool_name: str
    arguments: dict[str, object]
    expires_at: str


@dataclass(slots=True)
class _PendingApproval:
    owner_user_id: str
    future: asyncio.Future[ApprovalDecision]
    expires_at: datetime
    view: ToolApprovalView
```

Protect the dictionary with `asyncio.Lock`. `request()` inserts once, awaits
`asyncio.wait_for(asyncio.shield(future), timeout)`, resolves expiry, and removes
its own entry in `finally`. `decide()` checks ID, owner, expiry, and completion
before setting the future. Increment process-local counters at each lifecycle
transition, including cancellation and broker errors. Log only ID, tool name,
decision, and duration.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest tests/unit/servers/web/test_tool_approval_broker.py -v
git add src/internal/servers/web/tool_approval.py tests/unit/servers/web/test_tool_approval_broker.py
git commit -m "feat: add in-process tool approval broker"
```

---

### Task 4: Wire authenticated approvals through FastAPI and SSE

**Files:**
- Modify: `src/internal/configs/app_configs.py`
- Modify: `src/internal/servers/web/app.py`
- Modify: `tests/unit/test_configs.py`
- Modify: `tests/unit/servers/web/test_sse_streaming.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Produces: `AppSettings.tool_approval_timeout_seconds: float = 60.0`.
- Produces: `POST /api/agent/approvals/{approval_id}`.
- Produces: `approval_required` SSE event.
- Extends: `_run_agent_impl` and `_run_auto_routed` with `on_approval=None`.

- [ ] **Step 1: Write failing settings tests**

```python
def test_tool_approval_timeout_defaults_to_sixty_seconds() -> None:
    assert load_app_settings({}).tool_approval_timeout_seconds == 60.0


def test_tool_approval_timeout_reads_environment() -> None:
    settings = load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"})
    assert settings.tool_approval_timeout_seconds == 12.5
```

- [ ] **Step 2: Write failing SSE and endpoint tests**

Install a fake side-effecting tool and fake model response. Start the stream in
a thread, wait for `approval_required`, POST approve, and assert the same stream
later yields answer/done and the tool ran once. Add deny, timeout, anonymous,
wrong-user, unknown-ID, conflict, and disconnect cases. Assert every fail-closed
path executes zero side effects.

- [ ] **Step 3: Verify RED**

```bash
pytest tests/unit/test_configs.py tests/unit/servers/web/test_sse_streaming.py tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: failure for missing setting, callback, event, and endpoint.

- [ ] **Step 4: Implement settings, API views, and broker initialization**

Load positive float `TOOL_APPROVAL_TIMEOUT_SECONDS`, default `60.0`. Add:

```python
class ToolApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "deny"]


class ToolApprovalDecisionResponse(BaseModel):
    id: str
    decision: str
```

Create one broker in `create_web_app()` and store it on `app.state`. Map broker
errors to `403`, `404`, `409`, and `410`.

- [ ] **Step 5: Thread the callback through both tool-loop paths**

Add `on_approval=None` to `_run_agent_impl()` and `_run_auto_routed()`. Pass it
to automatic Tier-1 and explicit `tool_agent` loop calls only.

In `stream_agent`, resolve the authenticated user. If absent, pass no callback.
Otherwise build a callback that sanitizes arguments, registers with the broker,
queues `{"type": "approval_required", "approval": asdict(view)}`, and waits.
Cancellation must clean the entry. Keep `/api/agent` callback-free and add a
regression proving it never hangs.

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest tests/unit/test_configs.py tests/unit/servers/web/test_sse_streaming.py tests/unit/servers/web/test_web_experience_app.py tests/unit/test_execution_fallbacks.py -v
git add src/internal/configs/app_configs.py src/internal/servers/web/app.py tests/unit/test_configs.py tests/unit/servers/web/test_sse_streaming.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat: stream tool approval requests"
```

Expected: PASS.

---

### Task 5: Add frontend approval contracts and card

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`
- Create: `web/src/components/ToolApprovalCard.tsx`
- Create: `web/src/components/__tests__/ToolApprovalCard.test.tsx`
- Modify: `web/src/__tests__/api.test.ts`
- Modify: `web/src/styles.css`

**Interfaces:**
- Produces: `ToolApprovalView`, `SSEApprovalRequiredEvent`, `submitToolApproval`.
- Produces: `<ToolApprovalCard approval onDecision />`.

- [ ] **Step 1: Write failing API and card tests**

Assert `submitToolApproval("a1", "approve")` POSTs
`{"decision":"approve"}` to `/api/agent/approvals/a1`. Test card tool name,
safe arguments, countdown, Approve/Deny callbacks, locked buttons while pending
and after success, and inline endpoint errors.

- [ ] **Step 2: Verify RED**

```bash
cd web && npm run test:unit -- src/__tests__/api.test.ts src/components/__tests__/ToolApprovalCard.test.tsx
```

Expected: missing symbols/component failure.

- [ ] **Step 3: Add contracts and API helper**

```typescript
export interface ToolApprovalView {
  id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  expires_at: string;
}

export interface SSEApprovalRequiredEvent {
  type: "approval_required";
  approval: ToolApprovalView;
}
```

Add the event to `SSEEvent`. Implement `submitToolApproval` with `requestJson`,
POST, JSON body, same-origin credentials, and optional abort signal.

- [ ] **Step 4: Implement the focused card and styles**

Render an accessible region named `Approval required for <tool>`, a definition
list of the bounded server summary, countdown, and Approve/Deny buttons. Use
local `idle | submitting | decided | error` state; never render nested values as
HTML.

- [ ] **Step 5: Verify GREEN and commit**

```bash
cd web && npm run test:unit -- src/__tests__/api.test.ts src/components/__tests__/ToolApprovalCard.test.tsx
cd web && npm run typecheck
git add web/src/types.ts web/src/api.ts web/src/components/ToolApprovalCard.tsx web/src/components/__tests__/ToolApprovalCard.test.tsx web/src/__tests__/api.test.ts web/src/styles.css
git commit -m "feat: add tool approval card"
```

Expected: PASS.

---

### Task 6: Integrate approval cards into the live App stream

**Files:**
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Consumes: Task 5 event, API helper, and card.
- Preserves: one active `streamAgent()` iterator and existing event handling.

- [ ] **Step 1: Write failing App flow tests**

Mock a stream that yields `approval_required`, waits for a click, then yields
answer/done. Assert the card appears, Approve or Deny posts once, the answer
arrives, and `streamAgent` was called once. Add submission-error coverage.

- [ ] **Step 2: Verify RED**

Run: `cd web && npm run test:unit -- src/components/__tests__/App.test.tsx`

Expected: failure because App ignores approval events.

- [ ] **Step 3: Implement App integration**

Maintain `pendingApprovals: ToolApprovalView[]`. Upsert by ID on the SSE event,
render cards beside live progress, and post decisions with the current request
signal without aborting/restarting the stream. Clear approval state on new
search, new session, completion, and error.

- [ ] **Step 4: Verify GREEN and commit**

```bash
cd web && npm run test:unit -- src/components/__tests__/App.test.tsx src/components/__tests__/ToolApprovalCard.test.tsx
cd web && npm run typecheck
git add web/src/App.tsx web/src/components/__tests__/App.test.tsx
git commit -m "feat: approve tools during live agent runs"
```

Expected: PASS.

---

### Task 7: Verify compatibility, privacy, and phase boundary

**Files:**
- Modify: tests only if verification exposes a missing regression assertion.

**Interfaces:**
- Verifies: complete Phase 1 acceptance criteria.

- [ ] **Step 1: Scan approval payloads and logs**

```bash
rg -n "approval_required|ToolApprovalView|sanitize_tool_arguments|logger\..*arguments" src web tests
```

Expected: SSE uses only sanitized view arguments; no lifecycle logger receives
raw parsed arguments.

- [ ] **Step 2: Prove durable scope did not leak in**

```bash
git diff main...HEAD -- src/internal/db
rg -n "resume_token|restart.safe|durable approval" src web
```

Expected: no database diff and no production durable-resume contract.

- [ ] **Step 3: Run focused backend verification**

```bash
pytest tests/unit/test_tool_approval.py tests/unit/test_tool_registry.py tests/unit/test_api_tools.py tests/unit/test_on_turn_callback.py tests/unit/test_state_models.py tests/unit/servers/web/test_tool_approval_broker.py tests/unit/servers/web/test_sse_streaming.py tests/unit/servers/web/test_web_experience_app.py tests/unit/test_execution_fallbacks.py -v
```

Expected: PASS.

- [ ] **Step 4: Run frontend verification**

```bash
cd web && npm test -- --run
cd web && npm run typecheck
```

Expected: PASS.

- [ ] **Step 5: Run static and full backend checks**

```bash
ruff check src/agents src/tools src/internal/servers/web src/internal/configs tests/unit
ruff format --check src/agents src/tools src/internal/servers/web src/internal/configs tests/unit
git diff --check
pytest
```

Expected: all checks pass; integration tests requiring external services remain
outside the default suite.

- [ ] **Step 6: Commit verification-only adjustments if needed**

```bash
git add src web tests
git commit -m "test: verify tool approval integration"
```

If no files changed, do not create an empty commit.

## Final Acceptance Checklist

- [ ] Side-effecting and unspecified calls never execute before approval.
- [ ] Read-only tools preserve current behavior.
- [ ] Approval is per invocation and restricted to the initiating authenticated user.
- [ ] Denial, timeout, anonymity, cancellation, and broker failure fail closed.
- [ ] Skipped calls return to the model without masquerading as failures.
- [ ] Parallel calls wait for all decisions before any execution.
- [ ] SSE continues on the original connection after a decision.
- [ ] Browser and logs receive only sanitized summaries.
- [ ] Auto Tier-1 and explicit `tool_agent` share the approval seam.
- [ ] Non-streaming callers never hang.
- [ ] No persistence or restart-safe behavior is implemented or implied.
- [ ] All focused, full, frontend, type, lint, format, and diff checks pass.
