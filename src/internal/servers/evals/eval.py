from __future__ import annotations

import logging
import time
from typing import Any

from src.internal.chat.models import ChatFullResponse
from src.internal.servers.evals.models import ChatFullEvalResult
from src.internal.servers.evals.models import EvalationAck
from src.internal.servers.evals.models import EvalConfigurationOptions
from src.internal.servers.evals.models import EvalProvider
from src.internal.servers.evals.models import EvalTimings
from src.internal.servers.evals.models import EvalToolResult
from src.internal.servers.evals.models import MultiTurnEvalResult
from src.internal.servers.evals.models import ToolAssertion
from src.internal.servers.evals.provider import get_provider

logger = logging.getLogger(__name__)


def _chat_full_response_to_eval_result(
    full: ChatFullResponse,
    stream_start_time: float,
) -> ChatFullEvalResult:
    tools_called = [tc.tool_name for tc in full.tool_calls]
    tool_call_details: list[dict[str, Any]] = [
        {"tool_name": tc.tool_name, "tool_arguments": tc.tool_arguments}
        for tc in full.tool_calls
    ]
    total_ms = (time.time() - stream_start_time) * 1000
    return ChatFullEvalResult(
        answer=full.answer,
        tools_called=tools_called,
        tool_call_details=tool_call_details,
        citations=full.citation_info,
        timings=EvalTimings(
            total_ms=total_ms,
            llm_first_token_ms=None,
            tool_execution_ms={},
            stream_processing_ms=total_ms,
        ),
    )


def evaluate_tool_assertions(
    tools_called: list[str],
    assertions: ToolAssertion | None,
) -> tuple[bool | None, str | None]:
    if assertions is None:
        return None, None

    expected = set(assertions.expected_tools)
    called = set(tools_called)

    if assertions.require_all:
        missing = expected - called
        if missing:
            return False, (
                f"Missing expected tools: {sorted(missing)}. Called: {sorted(called)}"
            )
        return True, (
            f"All expected tools called: {sorted(expected)}. Called: {sorted(called)}"
        )

    matched = expected & called
    if not matched:
        return False, (
            f"None of expected tools called. Expected one of: {sorted(expected)}. Called: {sorted(called)}"
        )
    return True, (
        f"Expected tool(s) called: {sorted(matched)}. Called: {sorted(called)}"
    )


def _get_answer_with_tools(
    eval_input: dict[str, Any],
    configuration: EvalConfigurationOptions,
) -> EvalToolResult:
    """Run a single-turn eval against the live chat system.

    Requires a working DB connection (get_sqlalchemy_engine, create_chat_session,
    get_user_by_email). Raises NotImplementedError until wired to the repo DB layer.
    """
    raise NotImplementedError(
        "_get_answer_with_tools requires src.internal.db — wire before use"
    )


def _get_multi_turn_answer_with_tools(
    eval_input: dict[str, Any],
    configuration: EvalConfigurationOptions,
) -> MultiTurnEvalResult:
    """Run a multi-turn eval against the live chat system.

    Requires a working DB connection. Raises NotImplementedError until wired.
    """
    raise NotImplementedError(
        "_get_multi_turn_answer_with_tools requires src.internal.db — wire before use"
    )


def run_eval(
    configuration: EvalConfigurationOptions,
    data: list[dict[str, Any]] | None = None,
    remote_dataset_name: str | None = None,
    provider: EvalProvider = get_provider(),
) -> EvalationAck:
    if data is not None and remote_dataset_name is not None:
        raise ValueError("Cannot specify both data and remote_dataset_name")
    if data is None and remote_dataset_name is None:
        raise ValueError("Must specify either data or remote_dataset_name")

    return provider.eval(
        task=lambda eval_input: _get_answer_with_tools(eval_input, configuration),
        configuration=configuration,
        data=data,
        remote_dataset_name=remote_dataset_name,
        multi_turn_task=lambda eval_input: _get_multi_turn_answer_with_tools(
            eval_input, configuration
        ),
    )
