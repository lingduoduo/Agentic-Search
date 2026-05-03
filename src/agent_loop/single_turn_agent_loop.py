"""Single-turn agent loop implementation."""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from .agent_loop import AgentLoopBase, AgentLoopOutput, register, simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "WARN")))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that performs a single generation step."""

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.build_prompt_ids(messages)

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.generate_response_ids(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        response_mask = self.build_response_mask(response_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=1,
            metrics=metrics,
            request_id=request_id,
        )
