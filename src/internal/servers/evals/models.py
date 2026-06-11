from __future__ import annotations

import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from src.internal.llm.override_models import LLMOverride
from src.internal.servers.query_and_chat.streaming_models import CitationInfo

# Known built-in tool type names — override via EVAL_BUILTIN_TOOL_TYPES env var.
_DEFAULT_BUILTIN_TOOL_TYPES: list[str] = [
    t.strip()
    for t in os.environ.get(
        "EVAL_BUILTIN_TOOL_TYPES", "SearchTool,InternetSearchTool"
    ).split(",")
    if t.strip()
]


class ToolAssertion(BaseModel):
    expected_tools: list[str]
    require_all: bool = False


class EvalTimings(BaseModel):
    total_ms: float
    llm_first_token_ms: float | None = None
    tool_execution_ms: dict[str, float] = Field(default_factory=dict)
    stream_processing_ms: float | None = None


class ChatFullEvalResult(BaseModel):
    answer: str
    tools_called: list[str]
    tool_call_details: list[dict[str, Any]]
    citations: list[CitationInfo]
    timings: EvalTimings


class EvalToolResult(BaseModel):
    answer: str
    tools_called: list[str]
    tool_call_details: list[dict[str, Any]]
    citations: list[CitationInfo]
    assertion_passed: bool | None = None
    assertion_details: str | None = None
    timings: EvalTimings | None = None


class EvalMessage(BaseModel):
    message: str
    expected_tools: list[str] = Field(default_factory=list)
    require_all_tools: bool = False
    model: str | None = None
    model_provider: str | None = None
    temperature: float | None = None
    force_tools: list[str] = Field(default_factory=list)


class MultiTurnEvalResult(BaseModel):
    turn_results: list[EvalToolResult]
    all_passed: bool
    pass_count: int
    fail_count: int
    total_turns: int


class EvalConfiguration(BaseModel):
    llm: LLMOverride = Field(default_factory=LLMOverride)
    search_permissions_email: str
    allowed_tool_ids: list[int]


class EvalConfigurationOptions(BaseModel):
    builtin_tool_types: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BUILTIN_TOOL_TYPES)
    )
    llm: LLMOverride = LLMOverride(
        model_provider=None,
        model_version="gpt-4o",
        temperature=0.0,
    )
    search_permissions_email: str
    dataset_name: str
    no_send_logs: bool = False
    braintrust_project: str | None = None
    experiment_name: str | None = None

    def get_configuration(self, db_session: object) -> EvalConfiguration:
        raise NotImplementedError(
            "get_configuration requires src.internal.db — use a repo-local tool registry instead"
        )


class EvalationAck(BaseModel):
    success: bool


class EvalProvider(ABC):
    @abstractmethod
    def eval(
        self,
        task: Callable[[dict[str, Any]], EvalToolResult],
        configuration: EvalConfigurationOptions,
        data: list[dict[str, Any]] | None = None,
        remote_dataset_name: str | None = None,
        multi_turn_task: Callable[[dict[str, Any]], MultiTurnEvalResult] | None = None,
    ) -> EvalationAck:
        pass
