"""Helpers for turning search-agent trajectories into supervised examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentLoopOutput


@dataclass(frozen=True)
class SFTExample:
    """A supervised training example derived from one search-agent rollout."""

    prompt_messages: list[dict[str, Any]]
    completion: str
    trajectory_messages: list[dict[str, Any]]


def build_search_sft_example(
    input_messages: list[dict[str, Any]],
    output: AgentLoopOutput,
    *,
    include_environment_messages: bool = False,
) -> SFTExample:
    """Create an SFT example from a search-agent rollout.

    By default, the completion is the full assistant action trace, e.g.
    ``<plan>...<searches>...<fetch>...<answer>...`` joined across turns.

    When ``include_environment_messages`` is true, the returned
    ``trajectory_messages`` includes the full multi-turn conversation after the
    original prompt so a downstream trainer can reconstruct the whole dialogue.
    """
    completion = output.action_trace or output.final_answer or ""
    if not completion:
        raise ValueError("Search rollout does not contain an assistant action trace.")

    if include_environment_messages:
        trajectory_messages = list(output.trajectory_messages)
    else:
        trajectory_messages = [
            {"role": "assistant", "content": completion},
        ]

    return SFTExample(
        prompt_messages=list(input_messages),
        completion=completion,
        trajectory_messages=trajectory_messages,
    )
