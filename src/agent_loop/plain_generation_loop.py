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
    simple_timer,
)


@dataclass(frozen=True)
class PlainGenerationLoopConfig(AgentLoopConfig):
    """Configuration for one-shot model generation with no tools or retrieval."""


@register("plain_generation")
class PlainGenerationLoop(AgentLoopBase):
    """Simplest smoke-test loop: prompt -> model -> text response."""

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        config: AgentLoopConfig | PlainGenerationLoopConfig | None = None,
        loop: Any | None = None,
    ) -> None:
        cfg = config or PlainGenerationLoopConfig()
        super().__init__(
            tokenizer=tokenizer,
            server_manager=server_manager,
            config=AgentLoopConfig(
                prompt_length=cfg.prompt_length,
                response_length=cfg.response_length,
            ),
            loop=loop,
        )

    def _decode(self, token_ids: list[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return ""

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        request_id = uuid4().hex

        with simple_timer("generate_sequences", metrics):
            prompt_ids = await self.build_prompt_ids(messages)
            response_ids = await self.generate_response_ids(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

        answer = self._decode(response_ids)
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
