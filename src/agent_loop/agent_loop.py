"""Repo-native agent loop helpers for search-oriented generation."""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

_REGISTERED_AGENT_LOOPS: dict[str, type["AgentLoopBase"]] = {}


def register(name: str):
    """Register an agent loop class by name."""

    def decorator(cls: type["AgentLoopBase"]) -> type["AgentLoopBase"]:
        _REGISTERED_AGENT_LOOPS[name] = cls
        return cls

    return decorator


def get_registered_agent_loop(name: str) -> type["AgentLoopBase"]:
    """Return a registered agent loop class by name."""

    try:
        return _REGISTERED_AGENT_LOOPS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown agent loop: {name!r}") from exc


def list_registered_agent_loops() -> list[str]:
    """List registered agent loop names."""

    return sorted(_REGISTERED_AGENT_LOOPS)


@dataclass(frozen=True)
class AgentLoopConfig:
    prompt_length: int = 4096
    response_length: int = 512


@dataclass
class AgentLoopOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    metrics: dict[str, float] = field(default_factory=dict)
    request_id: str | None = None
    context: Any | None = None  # AgentContext when produced by SearchAgentLoop


@contextmanager
def simple_timer(name: str, metrics: dict[str, float]):
    """Small timing helper for per-step metrics."""

    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[name] = time.perf_counter() - start


class AgentLoopBase:
    """Minimal async agent loop base used inside this repository."""

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        config: AgentLoopConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.server_manager = server_manager
        self.config = config or AgentLoopConfig()
        self.loop = loop
        self.prompt_length = self.config.prompt_length
        self.response_length = self.config.response_length

    async def get_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        return self.loop

    async def build_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize a chat message list using the best tokenizer API available."""

        event_loop = await self.get_loop()
        return await event_loop.run_in_executor(None, lambda: self._build_prompt_ids_sync(messages))

    def _build_prompt_ids_sync(self, messages: list[dict[str, Any]]) -> list[int]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            return list(prompt_ids)[-self.prompt_length :]

        joined = "\n".join(message.get("content", "") for message in messages)
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(joined))[-self.prompt_length :]
        raise TypeError("tokenizer must implement apply_chat_template(...) or encode(...).")

    async def generate_response_ids(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str | None = None,
    ) -> list[int]:
        request_id = request_id or uuid4().hex
        generate_result = self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        response_ids = await generate_result if inspect.isawaitable(generate_result) else generate_result
        return list(response_ids)[: self.response_length]

    def build_response_mask(self, response_ids: list[int]) -> list[int]:
        return [1] * len(response_ids)

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        raise NotImplementedError
