# Generated Context Pack

# Tool Agent Human Approval

## Sources

- [Specification: 2026-06-27-tool-agent-human-approval-design.md](../specs/2026-06-27-tool-agent-human-approval-design.md)
- [Plan: 2026-06-27-tool-agent-human-approval.md](../plans/2026-06-27-tool-agent-human-approval.md)

## Specification Context

### Goal

Require the initiating user to approve each side-effecting `ToolAgentLoop`
invocation before it executes, while preserving the repository's current
in-memory agent loop and live SSE request model.

This phase adds a truthful in-process approval seam. It does not claim durable
pause or restart-safe resume; those require a persisted agent-run runtime and
belong in a separate design.

### Current architecture

The generic tool loop is request-local. `ToolAgentLoop.run()` keeps prompt IDs,
response masks, working messages, turn counters, and tool results in local
variables. Parsed tool calls are executed immediately and concurrently through
`_call_tool()`.

The web backend constructs a new loop for each automatic Tier-1 or explicit
`tool_agent` request. The non-streaming endpoint awaits the whole request. The
streaming endpoint runs the same coroutine in a task, forwards progress through
an in-memory queue, and persists the assistant message only after completion.

Several existing names suggest approval concepts but do not implement them:

- `Plan.requires_human_approval` has no runtime consumers.
- `EndpointPolicy.ASK` describes an absent external-app egress layer and has no
  call sites in this repository.
- notification constants for approval are declarations only.

The design therefore extends the working `Tool` and `ToolAgentLoop` paths. It
does not build on those inactive declarations.

### Scope

Phase 1 covers generic tools invoked by `ToolAgentLoop`, including the loop used
by automatic Tier-1 routing and explicit `tool_agent` mode.

Search-agent XML actions remain automated. Search and reranking are read-only
and use a different loop. Scheduled-task approval, external-app policy, direct
registry REST invocation, and MCP invocation are also outside this phase.

### Tool and loop tests

- function-tool unspecified defaults and explicit effect classification,
- OpenAPI HTTP-method inference,
- read-only calls execute without a callback,
- side-effecting calls fail closed without a callback,
- approval executes once,
- denial and timeout never call `Tool.execute`,
- skipped calls are returned to the model and do not count as failures,
- true tool failures preserve current stopping behavior,
- parallel approval requests block the whole batch until all resolve.

### Broker and web tests

- initiating-user ownership,
- approve, deny, timeout, conflict, unknown-ID, and expired-ID responses,
- disconnect and exception cleanup,
- approval event ordering before subsequent progress,
- automatic Tier-1 and explicit `tool_agent` callback wiring,
- non-streaming fail-closed behavior.

### Frontend tests

- approval card rendering and sanitised arguments,
- approve and deny submissions,
- button locking and endpoint errors,
- expiry display,
- stream continuation after a decision,
- unchanged behavior for streams without approval events.

The complete backend, frontend, typecheck, Ruff, formatting, and diff checks
must pass.

### Out of scope and Phase 2 boundary

Phase 1 does not provide:

- persistence across process restart,
- reconnectable approval streams,
- durable or exactly-once side-effect execution,
- remembered approvals,
- administrator approval,
- cross-process broker coordination,
- scheduled-task approval,
- external-app egress policy,
- approval for direct registry REST or MCP calls.

Phase 2 may introduce a persisted agent-run state machine, durable invocation
records, reconnectable event delivery, idempotency-key propagation, and
restart-safe resume. That work must be designed separately against the runtime
created here; Phase 1 must not imply those guarantees.

## Implementation Plan Context

### Global Constraints

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
