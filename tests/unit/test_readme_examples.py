"""Tests for runnable README examples."""

from __future__ import annotations

import asyncio

from examples.search_agent_loop_example import run_search_agent_loop_example
from src.agent_loop import SearchResult
from src.agent_loop.search_client import aiohttp


class DummyTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DummyServerManager:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict,
    ) -> list[int]:
        del request_id, prompt_ids, sampling_params
        response = self.responses[self.index]
        self.index += 1
        return [ord(char) for char in response]


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.closed = False

    async def retrieve(self, queries: list[str], topk: int | None = None):
        del topk
        self.calls.append(list(queries))
        return [
            [
                SearchResult(
                    contents=(
                        '"Dense vs sparse"\nDense retrieval uses vectors; '
                        "sparse retrieval uses lexical matching."
                    )
                )
            ]
            for _ in queries
        ]

    async def fetch_urls(self, urls: list[str]):
        del urls
        return []

    async def aclose(self) -> None:
        self.closed = True


def test_search_agent_loop_readme_example_runs_with_fake_backends():
    # SearchAgentLoop constructs a default SearchClient before the example swaps
    # in the fake client. Avoid requiring aiohttp for this pure unit smoke test.
    def client_timeout(total):
        del total
        return None

    aiohttp.ClientTimeout = client_timeout

    tokenizer = DummyTokenizer()
    search_client = FakeSearchClient()
    server_manager = DummyServerManager(
        [
            "<search>dense vs sparse retrieval</search>",
            (
                "<answer>Dense uses vectors; sparse uses lexical matching. "
                "[R1Q1D1]</answer>"
            ),
        ]
    )

    output = asyncio.run(
        run_search_agent_loop_example(
            tokenizer=tokenizer,
            server_manager=server_manager,
            search_client=search_client,
            max_turns=3,
            max_search_limit=1,
            sampling_params={"temperature": 0.0},
        )
    )

    assert (
        output.final_answer
        == "Dense uses vectors; sparse uses lexical matching. [R1Q1D1]"
    )
    assert output.metrics["search_rounds"] == 1.0
    assert search_client.calls == [["dense vs sparse retrieval"]]
