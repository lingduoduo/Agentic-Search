"""Generic multi-turn function/tool-calling agent loop.

The agent generates a response, the ToolParser extracts function calls from it,
the tools are executed in parallel, and their results are injected back as
``{"role": "tool", ...}`` messages before the next generation step.

This loop is intentionally not tied to search-agent XML actions. Search may be
registered as one function tool, but the same loop can call calculators,
databases, file tools, or any other JSON-schema-described function.

Supported tool-call formats are controlled by ``ToolAgentLoopConfig.tool_parser_format``:
    - ``"hermes"``  — NousResearch Hermes 2.5 / 3
    - ``"llama3"``  — Meta Llama 3.1 / 3.2
    - ``"json"``    — generic JSON fallback

Usage::

    from src import ToolAgentLoop, ToolAgentLoopConfig
    from src import FunctionTool

    @FunctionTool.from_fn(description="Search the web", parameters={...})
    async def search(query: str) -> str:
        ...

    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        tools=[search],
        config=ToolAgentLoopConfig(tool_parser_format="json"),
    )
    output = await loop.run(messages, sampling_params)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4

from src.agents.core.base import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    OnTurnCallback,
    register,
    simple_timer,
)
from src.agents.core.state import PerformanceMetrics, TaskStatus, ToolExecutionResult
from src.tools.base import Tool, ToolEffect
from src.tools.parsers import FunctionCall, ToolParser
from src.tools.validation import validate_arguments

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))


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


@dataclass(frozen=True)
class ToolAgentLoopConfig(AgentLoopConfig):
    """Configuration for ToolAgentLoop.

    Inherits ``prompt_length`` and ``response_length`` from AgentLoopConfig.
    """

    max_user_turns: int = 10
    max_assistant_turns: int = 10
    max_parallel_calls: int = 4
    max_tool_response_length: int = 2048
    # How to truncate a tool response that exceeds max_tool_response_length:
    #   "left"   — keep the start, append "...(truncated)"
    #   "right"  — prepend "(truncated)...", keep the end
    #   "middle" — keep equal halves from start and end
    tool_response_truncate_side: str = "right"
    tool_parser_format: str = "json"
    approval_timeout_seconds: float = 60.0


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    """Multi-turn agent loop that executes generic tool calls from the model.

    Each iteration:
      1. Tokenise the current message history (including tool schemas).
      2. Generate a response.
      3. Parse tool calls from the response with the configured ToolParser.
      4. If tool calls are found: execute them in parallel, append the results
         as ``{"role": "tool"}`` messages, and continue.
      5. Stop when no tool calls are found, a turn limit is reached, or the
         response budget is exhausted.

    ``response_mask`` is 1 for model-generated tokens and 0 for tool-response
    tokens injected back into the prompt — matching the VERL rollout convention.
    """

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        tools: list[Tool] | None = None,
        config: ToolAgentLoopConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        cfg = config or ToolAgentLoopConfig()
        super().__init__(
            tokenizer=tokenizer,
            server_manager=server_manager,
            config=cfg,
            loop=loop,
        )
        self.tool_config = cfg
        _tools = list(tools or [])
        self.tools: dict[str, Tool] = {t.name: t for t in _tools}
        self.tool_schemas: list[dict[str, Any]] = [t.schema.to_dict() for t in _tools]
        self.tool_parser: ToolParser = ToolParser.get_tool_parser(
            cfg.tool_parser_format, tokenizer
        )
        # Baseline token count produced by apply_chat_template for an empty
        # conversation — used to strip the template prefix when re-tokenising
        # tool responses so they don't double-count the system prompt.
        self._template_prefix_len: int = self._measure_template_prefix()

    def _measure_template_prefix(self) -> int:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return 0
        try:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                tokenize=True,
            )
            return len(ids)
        except Exception as exc:
            logger.debug("Could not measure template prefix length: %s", exc)
            return 0

    def _build_prompt_ids_with_tools_sync(
        self, messages: list[dict[str, Any]]
    ) -> list[int]:
        """Like _build_prompt_ids_sync but injects tool schemas into the template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas or None,
                add_generation_prompt=True,
                tokenize=True,
            )
            return list(ids)[-self.prompt_length :]
        # Fallback: no tool schema injection
        return self._build_prompt_ids_sync(messages)

    def _truncate_tool_response(self, text: str) -> str:
        limit = self.tool_config.max_tool_response_length
        if len(text) <= limit:
            return text
        side = self.tool_config.tool_response_truncate_side
        if side == "left":
            return text[:limit] + "...(truncated)"
        if side == "right":
            return "(truncated)..." + text[-limit:]
        half = limit // 2
        return text[:half] + "...(truncated)..." + text[-half:]

    async def _call_tool(self, tool_call: FunctionCall) -> ToolExecutionResult:
        """Execute one tool call and return a structured execution result."""
        tool = instance_id = None
        start = time.perf_counter()
        status = TaskStatus.FAILED
        result: Any = None
        error_code: str | None = None
        error_message: str | None = None
        elapsed = 0.0
        args = tool_call.parsed_arguments()
        try:
            tool = self.tools[tool_call.name]
            errors = validate_arguments(tool.schema.parameters, args)
            if errors:
                error_code = "invalid_arguments"
                error_message = "; ".join(errors)
            else:
                instance_id = await tool.create()
                result, _, _ = await tool.execute(instance_id, args)
                elapsed = time.perf_counter() - start
                status = TaskStatus.COMPLETED
                self._record_tool_stage(tool_call.name, args, result)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception("Error executing tool %r: %s", tool_call.name, exc)
            error_code = type(exc).__name__
            error_message = str(exc)
        finally:
            if tool is not None and instance_id is not None:
                await tool.release(instance_id)
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=status,
            result=result,
            arguments=args,
            performance=PerformanceMetrics(
                execution_time=elapsed,
                success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
            ),
            error_code=error_code,
            error_message=error_message,
        )

    async def _request_approval(
        self,
        tool_call: FunctionCall,
        on_approval: ToolApprovalCallback | None,
        metrics: dict[str, float],
    ) -> ApprovalDecision:
        tool = self.tools.get(tool_call.name)
        if tool is not None and tool.effect is ToolEffect.READ_ONLY:
            return ApprovalDecision.APPROVE

        metrics["tool_approvals_requested"] += 1
        if on_approval is None:
            metrics["tool_approvals_denied"] += 1
            return ApprovalDecision.DENY

        created_at = datetime.now(UTC)
        request = ToolApprovalRequest(
            approval_id=uuid4().hex,
            tool_name=tool_call.name,
            arguments=tool_call.parsed_arguments(),
            created_at=created_at,
            expires_at=created_at
            + timedelta(seconds=self.tool_config.approval_timeout_seconds),
        )
        try:
            decision = await asyncio.wait_for(
                on_approval(request), timeout=self.tool_config.approval_timeout_seconds
            )
        except asyncio.TimeoutError:
            metrics["tool_approvals_expired"] += 1
            return ApprovalDecision.EXPIRED
        except asyncio.CancelledError:
            metrics["tool_approvals_cancelled"] += 1
            raise
        except Exception:
            logger.exception("Approval callback failed for tool %r", tool_call.name)
            metrics["tool_approval_errors"] += 1
            metrics["tool_approvals_denied"] += 1
            return ApprovalDecision.DENY

        metrics[
            "tool_approvals_approved"
            if decision is ApprovalDecision.APPROVE
            else "tool_approvals_expired"
            if decision is ApprovalDecision.EXPIRED
            else "tool_approvals_denied"
        ] += 1
        return decision

    @staticmethod
    def _skipped_tool_result(
        tool_call: FunctionCall, decision: ApprovalDecision
    ) -> ToolExecutionResult:
        error_code = (
            "approval_expired"
            if decision is ApprovalDecision.EXPIRED
            else "approval_denied"
        )
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=TaskStatus.SKIPPED,
            result=None,
            arguments=tool_call.parsed_arguments(),
            performance=PerformanceMetrics(execution_time=0.0, success_rate=0.0),
            error_code=error_code,
            error_message="Tool execution skipped because approval was not granted.",
        )

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn: "OnTurnCallback | None" = None,
        on_approval: ToolApprovalCallback | None = None,
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {
            "tool_approvals_requested": 0,
            "tool_approvals_approved": 0,
            "tool_approvals_denied": 0,
            "tool_approvals_expired": 0,
            "tool_approvals_cancelled": 0,
            "tool_approval_errors": 0,
        }
        request_id = uuid4().hex
        event_loop = await self.get_loop()
        prompt_ids: list[int] = await event_loop.run_in_executor(
            None,
            lambda: self._build_prompt_ids_with_tools_sync(messages),
        )
        response_mask: list[int] = []
        working_messages: list[dict[str, Any]] = list(messages)
        tool_results: list[ToolExecutionResult] = []
        user_turns = 0
        assistant_turns = 0
        final_answer: str | None = None

        while True:
            # ── generate ─────────────────────────────────────────────────
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.generate_response_ids(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}_{assistant_turns}",
                )

            prompt_ids = prompt_ids + response_ids
            response_mask.extend([1] * len(response_ids))
            assistant_turns += 1

            # ── stopping conditions ───────────────────────────────────────
            if len(response_mask) >= self.response_length:
                break
            if (
                self.tool_config.max_assistant_turns
                and assistant_turns >= self.tool_config.max_assistant_turns
            ):
                break
            if (
                self.tool_config.max_user_turns
                and user_turns >= self.tool_config.max_user_turns
            ):
                break

            # ── parse tool calls ──────────────────────────────────────────
            assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(
                response_ids
            )
            working_messages.append({"role": "assistant", "content": assistant_content})
            final_answer = assistant_content
            if not tool_calls:
                break

            # Resolve the whole batch before any tool may execute.
            truncated_calls = tool_calls[: self.tool_config.max_parallel_calls]
            decisions = await asyncio.gather(
                *[
                    self._request_approval(tc, on_approval, metrics)
                    for tc in truncated_calls
                ]
            )

            # ── execute approved tools in parallel ────────────────────────
            with simple_timer("tool_calls", metrics):
                tool_execution_results = await asyncio.gather(
                    *[
                        self._call_tool(tc)
                        if decision is ApprovalDecision.APPROVE
                        else asyncio.sleep(
                            0, result=self._skipped_tool_result(tc, decision)
                        )
                        for tc, decision in zip(truncated_calls, decisions)
                    ]
                )
            tool_results.extend(tool_execution_results)
            if on_turn is not None:
                for r in tool_execution_results:
                    await on_turn(assistant_turns, r.tool_name, 0)

            if any(r.status is TaskStatus.FAILED for r in tool_execution_results):
                break

            tool_responses = [
                {
                    "role": "tool",
                    "content": self._truncate_tool_response(
                        str(result.result)
                        if result.status is TaskStatus.COMPLETED
                        else json.dumps(
                            {
                                "status": "skipped",
                                "error_code": result.error_code,
                            }
                        )
                    ),
                }
                for result in tool_execution_results
            ]
            working_messages.extend(tool_responses)

            # ── re-tokenise tool responses and append ─────────────────────
            tool_response_ids: list[int] = await event_loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    list(tool_responses), add_generation_prompt=True, tokenize=True
                ),
            )
            # Strip the template prefix to avoid re-including the system prompt tokens.
            tool_response_ids = tool_response_ids[self._template_prefix_len :]

            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids = prompt_ids + tool_response_ids
            response_mask.extend([0] * len(tool_response_ids))
            user_turns += 1

        # Split accumulated prompt_ids back into prompt / response portions.
        n = len(prompt_ids) - len(response_mask)
        final_prompt_ids = prompt_ids[:n]
        final_response_ids = prompt_ids[n:]

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            request_id=request_id,
            trajectory_messages=working_messages,
            action_trace="\n".join(
                json.dumps(r.to_dict(), default=str) for r in tool_results
            )
            or None,
            final_answer=final_answer,
        )
