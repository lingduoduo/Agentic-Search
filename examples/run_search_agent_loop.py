"""Minimal direct-use example for SearchAgentLoop.

The README points here instead of carrying a long code block. Tests exercise
this module with fake tokenizer/model/search backends so the public snippet
stays importable without loading a real model.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agent_loop import (
    SearchResult,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchEvaluationConfig,
)
from src.retrieval.client import aiohttp


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

    if search_client is not None:
        _allow_fake_search_client_without_aiohttp()

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


def _allow_fake_search_client_without_aiohttp() -> None:
    """Let fake-client examples construct SearchAgentLoop without aiohttp."""

    try:
        aiohttp.ClientTimeout
    except ModuleNotFoundError:
        aiohttp.ClientTimeout = _noop_client_timeout


def _noop_client_timeout(total):
    del total
    return None


class DemoTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DemoServerManager:
    def __init__(self) -> None:
        self.responses = [
            "<search>dense vs sparse retrieval</search>",
            (
                "<answer>Dense retrieval uses vector similarity, while sparse "
                "retrieval uses lexical term matching. [R1Q1D1]</answer>"
            ),
        ]
        self.index = 0

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        del request_id, prompt_ids, sampling_params
        response = self.responses[self.index]
        self.index += 1
        return [ord(char) for char in response]


class DemoSearchClient:
    async def retrieve(self, queries: list[str], topk: int | None = None):
        del topk
        return [
            [
                SearchResult(
                    contents=(
                        '"Dense vs sparse retrieval"\nDense retrieval represents '
                        "queries and documents with vectors; sparse retrieval "
                        "matches weighted lexical terms."
                    )
                )
            ]
            for _ in queries
        ]

    async def fetch_urls(self, urls: list[str]):
        del urls
        return []

    async def aclose(self) -> None:
        return None


async def run_demo() -> Any:
    return await run_search_agent_loop_example(
        tokenizer=DemoTokenizer(),
        server_manager=DemoServerManager(),
        search_client=DemoSearchClient(),
        max_turns=3,
        max_search_limit=1,
        sampling_params={"temperature": 0.0, "max_tokens": 128},
    )


def main() -> None:
    output = asyncio.run(run_demo())
    print(output.final_answer)
    print("Metrics:", output.metrics)


if __name__ == "__main__":
    main()
