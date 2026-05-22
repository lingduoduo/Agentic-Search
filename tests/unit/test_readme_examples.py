"""Tests for runnable README examples."""

from __future__ import annotations

import asyncio

from examples.run_grpo_training_pipeline import run_demo as run_grpo_demo
from examples.run_search_agent_loop import run_search_agent_loop_example
from examples.run_search_pipeline import AccessPolicy
from examples.run_search_pipeline import InMemorySearchIndex
from examples.run_search_pipeline import SearchDocument
from examples.run_search_pipeline import SearchFilters
from examples.run_search_pipeline import SearchRequest
from examples.run_search_pipeline import SearchUser
from examples.run_search_pipeline import run_demo as run_pipeline_demo
from examples.run_search_pipeline import search_pipeline
from examples.run_search_trace_workflow import run_sft_demo, run_workflow_demo
from src import SearchResult


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


def test_search_trace_workflow_example_renders_screenshot_style_trace():
    output, trace, search_client = asyncio.run(run_workflow_demo())

    assert output.final_answer == "John William Henry II"
    assert output.metrics["search_rounds"] == 3.0
    assert search_client.calls == [
        ["Jed Hoyer or John William Henry II"],
        ["John William Henry II"],
        ["Jed Hoyer birth year"],
    ]
    assert "Question: Who is older, Jed Hoyer or John William Henry II?" in trace
    assert "Ground Truth: John William Henry II" in trace
    assert "<think>I need to determine" in trace
    assert "<search>John William Henry II</search>" in trace
    assert "<information>" in trace
    assert "<answer>John William Henry II</answer>" in trace


def test_build_search_sft_example_script_uses_full_trace():
    example = asyncio.run(run_sft_demo())

    assert example.prompt_messages == [
        {
            "role": "user",
            "content": "Who is older, Jed Hoyer or John William Henry II?",
        }
    ]
    assert "<search>John William Henry II</search>" in example.completion
    assert "<information>" not in example.completion
    assert example.completion.endswith("<answer>John William Henry II</answer>")


def test_grpo_training_pipeline_example_runs_without_model_backends():
    result = run_grpo_demo()

    assert [round(value, 2) for value in result["rewards"]] == [0.96, 0.64, -0.08]
    assert len(result["advantages"]) == 3
    assert result["reward_components"][0]["correctness"] == 1.0
    assert "policy_loss" in result


def test_search_pipeline_example_applies_filters_and_permissions():
    sections = run_pipeline_demo()

    assert [section.document_id for section in sections] == ["public-guide"]
    assert sections[0].title == "Dense Retrieval Guide"

    index = InMemorySearchIndex(
        [
            SearchDocument(
                id="public",
                title="Public Doc",
                contents="rerank deployment notes",
                document_set="docs",
            ),
            SearchDocument(
                id="private",
                title="Private Doc",
                contents="rerank deployment secret",
                document_set="ops",
                access=AccessPolicy(
                    public=False,
                    allowed_group_ids=frozenset({"search-admins"}),
                ),
            ),
        ]
    )
    reader = SearchUser(id="reader", email="reader@example.test")
    admin = SearchUser(id="admin", group_ids=frozenset({"search-admins"}))

    reader_sections = search_pipeline(
        request=SearchRequest(query="rerank deployment", limit=10),
        index=index,
        user=reader,
    )
    admin_sections = search_pipeline(
        request=SearchRequest(query="rerank deployment", limit=10),
        index=index,
        user=admin,
    )
    filtered_sections = search_pipeline(
        request=SearchRequest(
            query="rerank deployment",
            filters=SearchFilters(document_set=frozenset({"ops"})),
            limit=10,
            bypass_acl=True,
        ),
        index=index,
        user=reader,
    )

    assert [section.document_id for section in reader_sections] == ["public"]
    assert {section.document_id for section in admin_sections} == {"public", "private"}
    assert [section.document_id for section in filtered_sections] == ["private"]


def test_search_pipeline_permission_filter_entrypoint_can_be_replaced():
    index = InMemorySearchIndex(
        [
            SearchDocument(
                id="blocked",
                title="Blocked",
                contents="dense retrieval",
                access=AccessPolicy(public=False),
            )
        ]
    )

    def allow_all(chunks, user, bypass_acl):
        del user, bypass_acl
        return list(chunks)

    sections = search_pipeline(
        request=SearchRequest(query="dense retrieval"),
        index=index,
        user=SearchUser(id="reader"),
        permission_filter=allow_all,
    )

    assert [section.document_id for section in sections] == ["blocked"]
