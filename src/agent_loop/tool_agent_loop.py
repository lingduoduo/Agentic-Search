"""Multi-turn tool-use agent loop.

The agent generates a response, the ToolParser extracts function calls from it,
the tools are executed in parallel, and their results are injected back as
``{"role": "tool", ...}`` messages before the next generation step.

Supported tool-call formats are controlled by ``ToolAgentLoopConfig.tool_parser_format``:
    - ``"hermes"``  — NousResearch Hermes 2.5 / 3
    - ``"llama3"``  — Meta Llama 3.1 / 3.2
    - ``"json"``    — generic JSON fallback

Usage::

    from src.agent_loop import ToolAgentLoop, ToolAgentLoopConfig
    from src.agent_loop.tool import FunctionTool

    @FunctionTool.from_fn(description="Search the web", parameters={...})
    async def search(query: str) -> str:
        ...

    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        tools=[search],
        config=ToolAgentLoopConfig(tool_parser_format="hermes"),
    )
    output = await loop.run(messages, sampling_params)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    register,
    simple_timer,
)
from .tool import Tool
from .tool_parser import FunctionCall, ToolParser

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))


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
    tool_parser_format: str = "hermes"


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    """Multi-turn agent loop that executes tool calls returned by the model.

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
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.tool_schemas: list[dict[str, Any]] = [
            t.schema.to_dict() for t in (tools or [])
        ]
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

    async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str] | Exception:
        """Execute one tool call and return a ``{"role": "tool", "content": ...}`` dict."""
        tool = instance_id = None
        try:
            tool = self.tools[tool_call.name]
            args = tool_call.parsed_arguments()
            instance_id = await tool.create()
            response_text, _, _ = await tool.execute(instance_id, args)
        except Exception as exc:
            logger.exception("Error executing tool %r: %s", tool_call.name, exc)
            return exc
        finally:
            if tool is not None and instance_id is not None:
                await tool.release(instance_id)

        return {
            "role": "tool",
            "content": self._truncate_tool_response(response_text),
        }

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        request_id = uuid4().hex
        event_loop = await self.get_loop()

        prompt_ids: list[int] = await event_loop.run_in_executor(
            None,
            lambda: self._build_prompt_ids_with_tools_sync(messages),
        )
        response_mask: list[int] = []
        user_turns = 0
        assistant_turns = 0

        while True:
            # ── generate ─────────────────────────────────────────────────
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.generate_response_ids(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}_{assistant_turns}",
                )

            prompt_ids = list(prompt_ids) + list(response_ids)
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
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            if not tool_calls:
                break

            # ── execute tools in parallel ─────────────────────────────────
            with simple_timer("tool_calls", metrics):
                tool_responses = await asyncio.gather(
                    *[
                        self._call_tool(tc)
                        for tc in tool_calls[: self.tool_config.max_parallel_calls]
                    ]
                )

            if any(isinstance(r, Exception) for r in tool_responses):
                break

            # ── re-tokenise tool responses and append ─────────────────────
            tool_response_ids: list[int] = await event_loop.run_in_executor(
                None,
                lambda resp=list(tool_responses): self.tokenizer.apply_chat_template(
                    resp, add_generation_prompt=True, tokenize=True
                ),
            )
            # Strip the template prefix to avoid re-including the system prompt tokens.
            tool_response_ids = tool_response_ids[self._template_prefix_len :]

            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids = list(prompt_ids) + list(tool_response_ids)
            response_mask.extend([0] * len(tool_response_ids))
            user_turns += 1

        # Split accumulated prompt_ids back into prompt / response portions.
        final_response_ids = prompt_ids[-len(response_mask) :]
        final_prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            request_id=request_id,
        )
