"""Shared runtime state models for agent orchestration.

These containers keep request, routing, planning, retrieval, and tool execution
state explicit without pulling in training dependencies. They are intentionally
small, slotted dataclasses because agent loops may create many of them during
rollout generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "TaskStatus",
    "ToolType",
    "TaskType",
    "UserRequest",
    "RetrievedDocument",
    "ToolCall",
    "ToolResult",
    "PlanStep",
    "Plan",
    "RouteDecision",
    "PerformanceMetrics",
    "ToolExecutionResult",
    "TaskNode",
    "AgentState",
]


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ToolType(Enum):
    """High-level tool category."""

    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    GENERATION = "generation"


class TaskType(Enum):
    """Execution task type kept for orchestration compatibility."""

    RETRIEVAL = "retrieval"
    ROUTING = "routing"
    PLANNING = "planning"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_GENERATION = "response_generation"


@dataclass(slots=True)
class UserRequest:
    user_id: str
    channel: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedDocument:
    doc_id: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanStep:
    step_id: str
    description: str
    needs_tool: bool = False
    tool_call: ToolCall | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Plan:
    workflow: str
    steps: list[PlanStep]
    requires_human_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    intent: str
    confidence: float
    should_retrieve: bool = True
    should_plan: bool = False
    should_use_tools: bool = False
    target_agent: str = "qa"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceMetrics:
    execution_time: float = 0.0
    cost_estimate: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    status: TaskStatus
    result: Any
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error_code: str | None = None
    error_message: str | None = None
    optimization_suggestions: list[str] = field(default_factory=list)
    retry_count: int = 0

    @property
    def success(self) -> bool:
        return self.status is TaskStatus.COMPLETED

    def to_tool_result(self) -> ToolResult:
        return ToolResult(
            tool_name=self.tool_name,
            success=self.success,
            output=self.result,
            error=self.error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskNode:
    task_id: str
    tool_name: str
    tool_type: ToolType
    params: dict[str, Any]
    dependencies: list[str]
    priority: int
    max_retries: int
    timeout: float
    is_critical: bool
    task_type: TaskType = TaskType.TOOL_EXECUTION
    status: TaskStatus = TaskStatus.PENDING
    result: ToolExecutionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentState:
    request_id: str
    user_request: UserRequest
    route: RouteDecision | None = None
    short_term_memory: list[dict[str, str]] = field(default_factory=list)
    long_term_memory: dict[str, Any] = field(default_factory=dict)
    retrieved_user_docs: list[RetrievedDocument] = field(default_factory=list)
    retrieved_policy_docs: list[RetrievedDocument] = field(default_factory=list)
    plan: Plan | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    draft_response: str | None = None
    final_response: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def record_trace(self, event: str, **payload: Any) -> None:
        self.trace.append({"event": event, **payload})

    def add_tool_result(self, result: ToolResult | ToolExecutionResult) -> None:
        if isinstance(result, ToolExecutionResult):
            result = result.to_tool_result()
        self.tool_results.append(result)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
