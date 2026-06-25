"""Serving-side LLM backend: the ServerManager protocol, concrete managers, and
a factory selecting between them. The training-side LLMGenerationManager
(model/generation.py) is a separate concern and intentionally not here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ServerManager(Protocol):
    """The model boundary every agent loop calls: tokens in, tokens out."""

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]: ...
