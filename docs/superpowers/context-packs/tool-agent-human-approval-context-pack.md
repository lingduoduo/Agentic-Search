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

…

### Scope

Phase 1 covers generic tools invoked by `ToolAgentLoop`, including the loop used
by automatic Tier-1 routing and explicit `tool_agent` mode.

Search-agent XML actions remain automated. Search and reranking are read-only
and use a different loop. Scheduled-task approval, external-app policy, direct
registry REST invocation, and MCP invocation are also outside this phase.

## Implementation Plan Context

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

…

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

…

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

…

### Final Acceptance Checklist

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

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
