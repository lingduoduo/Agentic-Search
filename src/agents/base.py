"""Repo-native agent loop helpers for search-oriented generation."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

# Matches the first recognised action tag in a model response.
# Covers all tags used by SearchAgentLoop; callers can override via action_re.
_DEFAULT_ACTION_RE: re.Pattern[str] = re.compile(
    r"<(plan|search_decision|subquestions|searches|search|fetch|answer)>(.*?)</\1>",
    re.DOTALL,
)

_DEFAULT_TERMINAL_ACTIONS: frozenset[str] = frozenset({"answer"})

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


@dataclass(frozen=True)
class RolloutStep:
    """One state → action step in a policy rollout trajectory.

    Maps directly to the RL triple:
        state (prompt_ids) → policy.generate() → action (response_ids) → terminal/continue

    action_type is the XML tag the model emitted: "search", "answer", "fetch",
    "plan", etc.  is_terminal=True means the trajectory should stop after this step.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    action_type: str | None
    action_content: str
    is_terminal: bool


@dataclass
class AgentLoopOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    metrics: dict[str, float] = field(default_factory=dict)
    request_id: str | None = None
    group_id: str | None = None
    rollout_index: int | None = None
    context: Any | None = None  # AgentContext when produced by SearchAgentLoop
    trajectory_messages: list[dict[str, Any]] = field(default_factory=list)
    action_trace: str | None = None
    final_answer: str | None = None  # Content of the last <answer> tag, if any


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
        return await event_loop.run_in_executor(
            None, lambda: self._build_prompt_ids_sync(messages)
        )

    def _build_prompt_ids_sync(self, messages: list[dict[str, Any]]) -> list[int]:
        chat_template = getattr(self.tokenizer, "chat_template", "__missing__")
        if hasattr(self.tokenizer, "apply_chat_template") and chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = self.tokenizer.encode(prompt_text)
            return list(prompt_ids)[-self.prompt_length :]

        joined = "\n".join(message.get("content", "") for message in messages)
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(joined))[-self.prompt_length :]
        raise TypeError(
            "tokenizer must implement apply_chat_template(...) or encode(...)."
        )

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
        response_ids = (
            await generate_result
            if inspect.isawaitable(generate_result)
            else generate_result
        )
        return list(response_ids)[: self.response_length]

    def build_response_mask(self, response_ids: list[int]) -> list[int]:
        return [1] * len(response_ids)

    def decode_response_ids(self, response_ids: list[int]) -> str:
        """Decode model-generated response ids when the tokenizer supports it."""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return ""

    async def _generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        metrics: dict[str, float],
        metric_name: str,
        request_id: str,
        sampling_params: dict[str, Any],
    ) -> tuple[list[int], list[int], str]:
        with simple_timer(metric_name, metrics):
            prompt_ids = await self.build_prompt_ids(messages)
            response_ids = await self.generate_response_ids(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        return prompt_ids, response_ids, self.decode_response_ids(response_ids)

    async def generate_rollout_step(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        action_re: re.Pattern[str] = _DEFAULT_ACTION_RE,
        terminal_actions: frozenset[str] = _DEFAULT_TERMINAL_ACTIONS,
        request_id: str | None = None,
    ) -> RolloutStep:
        """Sample one policy action and classify it for the RL loop.

        Analogous to actor_rollout_wg.generate_sequences — the per-step call that
        drives the trajectory:
            state (prompt_ids) → generate → classify action → RolloutStep

        The caller decides whether to continue (is_terminal=False) or stop
        (is_terminal=True) based on the returned action_type.
        """
        response_ids = await self.generate_response_ids(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        response_text = self.decode_response_ids(response_ids)
        match = action_re.search(response_text)
        action_type = match.group(1) if match else None
        action_content = match.group(2).strip() if match else ""
        return RolloutStep(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            response_mask=self.build_response_mask(response_ids),
            action_type=action_type,
            action_content=action_content,
            is_terminal=action_type in terminal_actions if action_type else True,
        )

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        raise NotImplementedError
