"""Plain one-shot generation loop used by CLI ``--mode single``.

This loop intentionally has no retrieval client, XML action parser, or tool
schema. It is the smallest useful path for checking that a model backend can
tokenize a prompt and produce text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    register,
)


@dataclass(frozen=True)
class PlainGenerationLoopConfig(AgentLoopConfig):
    """Configuration for one-shot model generation with no tools or retrieval."""


@register("plain_generation")
class PlainGenerationLoop(AgentLoopBase):
    """Simplest smoke-test loop: prompt -> model -> text response."""

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        request_id = uuid4().hex
        prompt_ids, response_ids, answer = await self._generate_text(
            messages,
            metrics=metrics,
            metric_name="generate_sequences",
            request_id=request_id,
            sampling_params=sampling_params,
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=self.build_response_mask(response_ids),
            num_turns=1,
            metrics=metrics,
            request_id=request_id,
            trajectory_messages=messages + [{"role": "assistant", "content": answer}],
            final_answer=answer,
        )
