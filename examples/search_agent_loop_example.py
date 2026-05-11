"""Minimal direct-use example for SearchAgentLoop.

The README points here instead of carrying a long code block. Tests exercise
this module with fake tokenizer/model/search backends so the public snippet
stays importable without loading a real model.
"""

from __future__ import annotations

from typing import Any

from src.agent_loop import (
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchEvaluationConfig,
)


async def run_search_agent_loop_example(
    *,
    tokenizer: Any,
    server_manager: Any,
    question: str = "Compare dense vs sparse retrieval.",
    search_url: str = "http://localhost:8000/retrieve",
    topk: int = 5,
    max_turns: int = 8,
    max_search_limit: int = 6,
    sampling_params: dict[str, Any] | None = None,
    search_client: Any | None = None,
) -> Any:
    """Build and run SearchAgentLoop with README-friendly defaults."""

    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            search_url=search_url,
            topk=topk,
            max_turns=max_turns,
            max_search_limit=max_search_limit,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1,
                min_total_results=1,
            ),
        ),
    )
    if search_client is not None:
        await loop._search_client.aclose()
        loop._search_client = search_client

    return await loop.run(
        messages=[{"role": "user", "content": question}],
        sampling_params=sampling_params or {"temperature": 0.7},
    )
