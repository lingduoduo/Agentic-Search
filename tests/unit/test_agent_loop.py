"""Unit tests for src.agent_loop."""

import asyncio

import pytest

from src.agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    PlainGenerationLoop,
    PlainGenerationLoopConfig,
    RolloutStep,
    SearchEvaluationConfig,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchResultEvaluator,
    SearchContext,
    SearchResult,
    SingleTurnAgentLoop,
    SingleTurnAgentLoopConfig,
    ToolAgentLoopConfig,
    get_registered_agent_loop,
    list_registered_agent_loops,
    register,
)


class DummyTokenizerWithTemplate:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [11, 12, 13, 14, 15]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DummyTokenizerWithEncode:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DummyServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids
        self.calls = []
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            }
        )
        if self.response_ids and isinstance(self.response_ids[0], list):
            response = self.response_ids[self.index]
            self.index += 1
            return list(response)
        return list(self.response_ids)


class DummySyncServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids

    def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        return list(self.response_ids)


class ConcreteAgentLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        del messages, sampling_params
        raise NotImplementedError


def test_build_prompt_ids_uses_chat_template_when_available():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=3, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids([{"role": "user", "content": "hello"}])
    )
    assert prompt_ids == [13, 14, 15]


def test_build_prompt_ids_falls_back_to_encode():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=4, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids(
            [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}]
        )
    )
    assert len(prompt_ids) == 4


def test_plain_generation_loop_runs_one_model_generation():
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(tokenizer.encode("plain answer"))
    loop = PlainGenerationLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=PlainGenerationLoopConfig(response_length=64),
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is FAISS?"}],
            {"temperature": 0.0},
        )
    )

    assert output.num_turns == 1
    assert output.final_answer == "plain answer"
    assert output.context is None
    assert len(server_manager.calls) == 1


def test_search_agent_default_prompt_includes_training_template_boundaries():
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
    )
    prompt = loop.search_config.system_prompt
    assert prompt is not None
    assert "<think>" in prompt
    assert "<search>" in prompt
    assert "<information>" in prompt
    assert "<answer>" in prompt
    assert "Never write or fabricate this block" in prompt


def test_generate_response_ids_truncates_to_response_length():
    server_manager = DummyServerManager([1, 2, 3, 4, 5])
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=10, response_length=3),
    )
    response_ids = asyncio.run(
        loop.generate_response_ids([9, 9], {"temperature": 0.1}, request_id="req-1")
    )
    assert response_ids == [1, 2, 3]
    assert server_manager.calls[0]["request_id"] == "req-1"


# ── force_search=True (classic pre-retrieval RAG, backward-compat) ───────────


def test_single_turn_agent_loop_force_search_returns_expected_output():
    """force_search=True: retrieve unconditionally, then generate once."""
    server_manager = DummyServerManager([21, 22, 23, 24])
    loop = SingleTurnAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(
            prompt_length=4, response_length=2, force_search=True
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("hello",): [
                [SearchResult(contents='"Greeting"\nHello world evidence')],
            ],
        }
    )
    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.7})
    )
    assert output.response_ids == [21, 22]
    assert output.response_mask == [1, 1]
    assert output.num_turns == 1
    assert output.context.num_rounds == 1
    assert output.context.queries == ["hello"]
    assert output.final_answer is not None
    assert output.trajectory_messages[-1]["role"] == "assistant"
    assert "retrieve" in output.metrics
    assert "generate_sequences" in output.metrics
    assert output.request_id is not None


def test_single_turn_agent_loop_force_search_injects_evidence_into_prompt():
    """force_search=True: retrieved evidence must appear in the prompt."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager([tokenizer.encode("Answer with evidence")])
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64, force_search=True),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what happened",): [
                [SearchResult(contents='"Doc A"\nEvidence body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "what happened"}],
            {"temperature": 0.0},
        )
    )

    prompt_text = tokenizer.decode(output.prompt_ids, skip_special_tokens=False)
    assert "<information>" in prompt_text
    assert "Evidence body" in prompt_text
    assert output.context.num_results == 1


def test_single_turn_agent_loop_force_search_disabled_retrieval():
    """force_search=True but use_retrieval=False: no retrieval, one generation."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager([tokenizer.encode("Direct answer")])
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(
            response_length=64, force_search=True, use_retrieval=False
        ),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.0})
    )

    assert fake_client.calls == []
    assert output.context.num_rounds == 0
    assert output.num_turns == 1


# ── default mode: tool-augmented one-shot ─────────────────────────────────────


def test_single_turn_agent_loop_tool_augmented_search_then_answer():
    """Default mode: model emits <search>, retrieves, then emits <answer>."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            # First generation: model decides to search
            tokenizer.encode("<search>Nobel Prize Physics 2024</search>"),
            # Second generation: model answers with evidence
            tokenizer.encode("<answer>Hopfield and Hinton</answer>"),
        ]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64),
    )
    loop._search_client = FakeSearchClient(
        {
            ("Nobel Prize Physics 2024",): [
                [SearchResult(contents='"Nobel 2024"\nHopfield and Hinton won')],
            ],
        }
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "Who won the Nobel Prize in Physics 2024?"}],
            {"temperature": 0.8},
        )
    )

    assert output.num_turns == 2
    assert output.context.num_rounds == 1
    assert output.context.queries == ["Nobel Prize Physics 2024"]
    assert output.final_answer == "Hopfield and Hinton"
    assert "retrieve" in output.metrics
    assert "generate_sequences" in output.metrics
    assert "generate_sequences_2" in output.metrics
    assert output.metrics["search_rounds"] == 1.0
    # Trajectory must include: original user msg, assistant <search>, user <information>, assistant <answer>
    roles = [m["role"] for m in output.trajectory_messages]
    assert roles.count("assistant") >= 1


def test_single_turn_agent_loop_tool_augmented_direct_answer():
    """Default mode: model emits <answer> directly — no retrieval call."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [tokenizer.encode("<answer>Paris is the capital of France.</answer>")]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is the capital of France?"}],
            {"temperature": 0.0},
        )
    )

    assert fake_client.calls == []  # no retrieval
    assert output.num_turns == 1  # only one generation step
    assert output.context.num_rounds == 0
    assert output.final_answer == "Paris is the capital of France."
    assert output.metrics["search_rounds"] == 0.0


def test_single_turn_agent_loop_tool_augmented_search_tag_no_retrieval():
    """Default mode: model emits <search> but use_retrieval=False — falls through to direct answer."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [tokenizer.encode("<search>some query</search>")]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64, use_retrieval=False),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.0})
    )

    assert fake_client.calls == []  # retrieval skipped
    assert output.num_turns == 1  # no second generation step
    assert output.context.num_rounds == 0


def test_single_turn_agent_loop_tool_augmented_observation_in_second_prompt():
    """Default mode: after <search>, the second prompt contains <information>."""
    tokenizer = DummyTokenizerWithEncode()
    second_gen_prompt_ids: list[list[int]] = []

    class CapturingServerManager:
        call_count = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.call_count += 1
            if self.call_count == 1:
                return tokenizer.encode("<search>query</search>")
            second_gen_prompt_ids.append(list(prompt_ids))
            return tokenizer.encode("<answer>found it</answer>")

    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=CapturingServerManager(),
        config=SingleTurnAgentLoopConfig(response_length=128),
    )
    loop._search_client = FakeSearchClient(
        {
            ("query",): [
                [SearchResult(contents='"Source"\nKey evidence here')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "question"}], {"temperature": 0.7})
    )

    assert output.num_turns == 2
    # The second generation prompt must include the retrieved evidence
    second_prompt_text = tokenizer.decode(
        second_gen_prompt_ids[0], skip_special_tokens=False
    )
    assert "<information>" in second_prompt_text
    assert "Key evidence here" in second_prompt_text


def test_register_stores_class_by_name():
    @register("test_agent_loop")
    class RegisteredLoop(ConcreteAgentLoop):
        pass

    assert RegisteredLoop.__name__ == "RegisteredLoop"


def test_get_registered_agent_loop_returns_single_turn_loop():
    registered = get_registered_agent_loop("single_turn_agent")
    assert registered is SingleTurnAgentLoop


def test_get_registered_agent_loop_returns_plain_generation_loop():
    registered = get_registered_agent_loop("plain_generation")
    assert registered is PlainGenerationLoop


def test_list_registered_agent_loops_includes_single_turn():
    assert "single_turn_agent" in list_registered_agent_loops()
    assert "plain_generation" in list_registered_agent_loops()


def test_tool_agent_loop_defaults_to_generic_json_parser():
    assert ToolAgentLoopConfig().tool_parser_format == "json"


def test_get_registered_agent_loop_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown agent loop"):
        get_registered_agent_loop("missing_loop")


def test_generate_response_ids_supports_sync_server_manager():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummySyncServerManager([7, 8, 9, 10]),
        config=AgentLoopConfig(prompt_length=10, response_length=2),
    )
    response_ids = asyncio.run(
        loop.generate_response_ids([1, 2], {"temperature": 0.3}, request_id="req-sync")
    )
    assert response_ids == [7, 8]


def test_search_result_information_block_supports_citation_prefix():
    ctx = SearchContext(
        query="alpha",
        results=[SearchResult(contents='"Alpha"\nBeta', url="https://example.com")],
    )
    assert (
        ctx.to_information_block(citation_prefix="R1Q1D")
        == "[R1Q1D1] (Title: Alpha) Beta URL: https://example.com"
    )


def test_search_result_evaluator_marks_weak_results_as_insufficient():
    evaluator = SearchResultEvaluator(
        SearchEvaluationConfig(
            min_results_per_query=2,
            min_total_results=3,
            min_top_score=0.8,
            min_avg_score=0.7,
        )
    )
    evaluation = evaluator.evaluate_round(
        [
            SearchContext(
                query="alpha",
                results=[SearchResult(contents='"Alpha"\nbody', score=0.4)],
            )
        ]
    )

    assert evaluation.is_sufficient is False
    assert evaluation.total_results == 1
    assert "below minimum" in evaluation.to_feedback_block()


def test_search_result_evaluator_marks_strong_results_as_sufficient():
    evaluator = SearchResultEvaluator(
        SearchEvaluationConfig(
            min_results_per_query=1,
            min_total_results=2,
            min_top_score=0.8,
            min_avg_score=0.7,
        )
    )
    evaluation = evaluator.evaluate_round(
        [
            SearchContext(
                query="alpha",
                results=[
                    SearchResult(contents='"Alpha"\nbody', score=0.9),
                    SearchResult(contents='"Alpha 2"\nbody', score=0.8),
                ],
            )
        ]
    )

    assert evaluation.is_sufficient is True
    assert "Verdict: SUFFICIENT" in evaluation.to_feedback_block()


class FakeSearchClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.fetch_calls = []
        self.fetch_responses = {}
        self.closed = False

    async def retrieve(self, queries, topk=None):
        del topk
        self.calls.append(list(queries))
        return self.responses[tuple(queries)]

    async def fetch_urls(self, urls):
        self.fetch_calls.append(list(urls))
        return self.fetch_responses[tuple(urls)]

    async def aclose(self):
        self.closed = True


def test_search_agent_loop_supports_plan_parallel_search_and_research_rounds():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<plan>Compare two sources and validate with a follow-up search.</plan>"
        ),
        tokenizer.encode("<searches>\n- first query\n- second query\n</searches>"),
        tokenizer.encode("<searches><query>refined query</query></searches>"),
        tokenizer.encode("<answer>Final report [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query", "second query"): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
                [SearchResult(contents='"Doc B"\nBeta body')],
            ],
            ("refined query",): [
                [SearchResult(contents='"Doc C"\nGamma body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == [
        ["first query", "second query"],
        ["refined query"],
    ]
    assert output.context is not None
    assert output.context.num_rounds == 2
    assert output.context.num_searches == 3
    assert output.context.queries == ["first query", "second query", "refined query"]
    assert output.num_turns == 4


def test_search_agent_loop_injects_search_evaluation_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2,
                min_total_results=2,
                min_top_score=0.8,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.5)],
            ],
        }
    )

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<search_evaluation>" in second_prompt
    assert "Verdict: INSUFFICIENT" in second_prompt
    assert "keep searching" in second_prompt


def test_search_agent_loop_closes_search_client_after_run():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager([tokenizer.encode("<answer>Done</answer>")]),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "answer directly"}], {"temperature": 0.0})
    )

    assert fake_client.closed is True


def test_search_agent_loop_rejects_answer_before_any_search():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done after search</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<answer_feedback>" in second_prompt
    assert "Search first" in second_prompt
    assert output.num_turns == 3
    assert output.context.num_rounds == 1


def test_search_agent_loop_rejects_answer_when_latest_evidence_is_insufficient():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\nrefined query\n</searches>"),
        tokenizer.encode("<answer>Done after refinement</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2,
                min_total_results=2,
                min_top_score=0.8,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.5)],
            ],
            ("refined query",): [
                [
                    SearchResult(contents='"Doc B"\nBeta body', score=0.95),
                    SearchResult(contents='"Doc C"\nGamma body', score=0.85),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert "<answer_feedback>" in third_prompt
    assert "latest search evaluation was insufficient" in third_prompt
    assert output.num_turns == 4
    assert output.context.num_rounds == 2


def test_search_agent_loop_handles_plan_and_searches_in_same_response():
    """When a model emits <plan> and <searches> in a single response, both are
    processed in one turn — no wasted round-trip for the plan acknowledgement."""
    tokenizer = DummyTokenizerWithEncode()
    combined = "<plan>Quick plan.</plan><searches>\nalpha\nbeta\n</searches>"
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(max_turns=4),
    )
    loop._search_client = FakeSearchClient(
        {
            ("alpha", "beta"): [
                [SearchResult(contents='"A"\nbody a')],
                [SearchResult(contents='"B"\nbody b')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == [["alpha", "beta"]]
    assert output.context.num_rounds == 1
    assert output.context.num_searches == 2
    assert output.num_turns == 2


def test_search_agent_loop_can_fetch_full_page_content():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<fetch>https://example.com/a</fetch>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            ),
        ],
    }
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    assert "<full_page>" in third_prompt
    assert "Full page body" in third_prompt


def test_search_agent_loop_deduplicates_queries_and_urls_and_tracks_metrics():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nalpha\nalpha\n</searches>"),
        tokenizer.encode("<fetch>https://example.com/a, https://example.com/a</fetch>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            ),
        ],
    }
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert fake_client.calls == [["alpha"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    assert output.metrics["search_rounds"] == 1.0
    assert output.metrics["search_queries"] == 1.0
    assert output.metrics["fetched_pages"] == 1.0
    assert len(output.context.fetched_pages) == 1
    assert output.metrics["useful_fetched_pages"] == 1.0
    assert output.metrics["unnecessary_fetch_count"] == 0.0


def test_search_agent_loop_registers_subquestions_and_tracks_task_searches():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: identify the voice actor\nT2: identify the developer\n</subquestions>"
            "<searches>\n[T1] Alice David Lara Croft voice\n[T2] Lara Croft game developer\n</searches>"
        ),
        tokenizer.encode("<answer>Done [R1Q1D1] [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=2
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice", "Lara Croft game developer"): [
                [
                    SearchResult(
                        contents='"Voice"\nAlice David', url="https://example.com/voice"
                    )
                ],
                [
                    SearchResult(
                        contents='"Developer"\nCrystal Dynamics',
                        url="https://example.com/dev",
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.context.tasks == {
        "T1": "identify the voice actor",
        "T2": "identify the developer",
    }
    assert output.context.turns[0].task_id == "T1"
    assert output.context.turns[1].task_id == "T2"
    assert output.metrics["active_subquestions"] == 2.0
    assert output.metrics["subquestions_covered"] == 2.0
    assert output.metrics["subquestion_coverage_ratio"] == 1.0


def test_search_agent_loop_rejects_answer_when_a_subquestion_is_unresolved():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: identify the voice actor\nT2: identify the developer\n</subquestions>"
            "<searches>\n[T1] Alice David Lara Croft voice\n</searches>"
        ),
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\n[T2] Lara Croft game developer\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice",): [
                [
                    SearchResult(
                        contents='"Voice"\nAlice David', url="https://example.com/voice"
                    )
                ],
            ],
            ("Lara Croft game developer",): [
                [
                    SearchResult(
                        contents='"Developer"\nCrystal Dynamics',
                        url="https://example.com/dev",
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert "T2: identify the developer" in third_prompt
    assert "<answer_feedback>" in third_prompt
    assert output.context.num_rounds == 2


def test_search_agent_loop_reports_subquestion_coverage_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: find founding year\nT2: find headquarters\n</subquestions>"
            "<searches>\n[T1] company founding year\n</searches>"
        ),
        tokenizer.encode("<answer>Too early</answer>"),
        tokenizer.encode("<searches>\n[T2] company headquarters\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("company founding year",): [
                [SearchResult(contents='"Founded"\n1999')],
            ],
            ("company headquarters",): [
                [SearchResult(contents='"HQ"\nNew York')],
            ],
        }
    )

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<subquestions_feedback>" in second_prompt
    assert "Covered:" in second_prompt
    assert "Needs more evidence:" in second_prompt


def test_search_agent_loop_skips_repeated_queries_with_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha query"]]
    assert "Repeated search skipped" in third_prompt
    assert output.metrics["repeated_search_queries"] == 1.0


def test_search_agent_loop_enforces_search_limit():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>beta query</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_search_limit=1,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha query"]]
    assert "Search limit reached" in third_prompt
    assert output.metrics["search_limit_hits"] == 1.0


def test_search_agent_loop_tracks_budget_exhausted_without_answer():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>beta query</search>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            max_search_limit=1,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.final_answer is None
    assert output.metrics["search_budget_exhausted_without_answer"] == 1.0


def test_search_agent_loop_allows_direct_answer_before_search_when_enabled():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<search_decision>answer</search_decision><answer>Paris</answer>"
        ),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            allow_internal_knowledge_answer=True,
        ),
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is the capital of France?"}],
            {"temperature": 0.0},
        )
    )

    assert output.num_turns == 1
    assert output.context.num_rounds == 0
    assert output.metrics["direct_answers"] == 1.0
    assert output.metrics["answer_allowed"] == 1.0


def test_search_agent_loop_requests_search_after_search_decision():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search_decision>search</search_decision>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<decision_feedback>" in second_prompt
    assert "Issue a <search> or <searches> action next" in second_prompt


def test_search_agent_loop_prompts_for_decision_when_no_action_before_search():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("I am thinking but have not decided yet."),
        tokenizer.encode("<search_decision>search</search_decision>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<answer_feedback>" in second_prompt
    assert "Use <search_decision>answer</search_decision>" in second_prompt


def test_search_agent_loop_emits_search_quality_metrics():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nweak query\nstrong query\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1,
                min_total_results=2,
                min_top_score=0.8,
                require_scores=True,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("weak query", "strong query"): [
                [SearchResult(contents='"Weak"\nbody', score=0.2)],
                [
                    SearchResult(contents='"Strong"\nbody', score=0.95),
                    SearchResult(contents='"Strong 2"\nbody', score=0.9),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["search_quality_score"] == pytest.approx(0.5, abs=0.001)
    assert output.metrics["evidence_insufficient_rounds"] == 1.0
    assert output.metrics["final_evidence_sufficient"] == 0.0
    assert output.metrics["answer_when_evidence_insufficient"] == 1.0


def test_search_agent_loop_blocks_direct_answer_when_internal_knowledge_disabled():
    """allow_internal_knowledge_answer=False prevents bypassing the search gate."""
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<search_decision>answer</search_decision><answer>Paris</answer>"
        ),
        tokenizer.encode("<searches>\nalpha\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            allow_internal_knowledge_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("alpha",): [[SearchResult(contents='"Doc A"\nAlpha body')]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q"}], {"temperature": 0.0})
    )

    assert output.metrics["direct_answers"] == 0.0
    assert output.context.num_rounds == 1


def test_search_agent_loop_search_decision_with_searches_fires_search_not_decision_feedback():
    """<search_decision>search</search_decision> alongside <searches> should execute
    the search without injecting a decision_feedback observation."""
    tokenizer = DummyTokenizerWithEncode()
    combined = "<search_decision>search</search_decision><searches>\nalpha\n</searches>"
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("alpha",): [[SearchResult(contents='"Doc A"\nAlpha body')]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    second_prompt = "".join(chr(t) for t in loop.server_manager.calls[1]["prompt_ids"])
    assert "<decision_feedback>" not in second_prompt
    assert "<information>" in second_prompt
    assert output.context.num_rounds == 1
    assert output.num_turns == 2


def test_search_client_config_derives_fetch_url_from_retrieve_url():
    from src.agent_loop.search_client import SearchClientConfig

    cases = [
        ("http://localhost:8000/retrieve", "http://localhost:8000/fetch"),
        ("http://localhost:8000/retrieve/", "http://localhost:8000/fetch"),
        ("http://host:9000/api/retrieve", "http://host:9000/api/fetch"),
        ("http://host/other", "http://host/other/fetch"),
    ]
    for url, expected in cases:
        assert SearchClientConfig(url=url).get_fetch_url() == expected, (
            f"Failed for {url!r}"
        )


def test_search_agent_loop_processes_search_and_fetch_in_same_turn():
    """When the model emits <searches> and <fetch> in the same turn, both are
    executed and their results appear in a single observation message."""
    tokenizer = DummyTokenizerWithEncode()
    combined = (
        "<searches>\nalpha query\n</searches><fetch>https://example.com/a</fetch>"
    )
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ]
            ]
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            )
        ]
    }
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    # Both search and fetch fired in turn 0.
    assert fake_client.calls == [["alpha query"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    # Both observations are in the same injected user message (turn 1 prompt).
    second_prompt = "".join(chr(t) for t in loop.server_manager.calls[1]["prompt_ids"])
    assert "<information>" in second_prompt
    assert "<full_page>" in second_prompt
    assert output.num_turns == 2


# ---------------------------------------------------------------------------
# generate_rollout_step — explicit RL step: state → action → terminal/continue
# ---------------------------------------------------------------------------


def test_generate_rollout_step_classifies_search_action():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            tokenizer.encode("<search>who invented radar</search>")
        ),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[1, 2, 3],
            sampling_params={"temperature": 0.7},
        )
    )
    assert isinstance(step, RolloutStep)
    assert step.action_type == "search"
    assert step.action_content == "who invented radar"
    assert step.is_terminal is False
    assert step.prompt_ids == [1, 2, 3]
    assert step.response_mask == [1] * len(step.response_ids)


def test_generate_rollout_step_classifies_answer_as_terminal():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            tokenizer.encode("<answer>Watson and Watt</answer>")
        ),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[4, 5],
            sampling_params={"temperature": 0.0},
        )
    )
    assert step.action_type == "answer"
    assert step.action_content == "Watson and Watt"
    assert step.is_terminal is True


def test_generate_rollout_step_marks_no_action_as_terminal():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(tokenizer.encode("I have no idea.")),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[1],
            sampling_params={"temperature": 0.5},
        )
    )
    assert step.action_type is None
    assert step.action_content == ""
    assert step.is_terminal is True


def test_generate_rollout_step_accepts_custom_action_re_and_terminal_actions():
    import re

    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(tokenizer.encode("<tool>calculator</tool>")),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[7, 8],
            sampling_params={"temperature": 0.0},
            action_re=re.compile(r"<(tool)>(.*?)</\1>", re.DOTALL),
            terminal_actions=frozenset({"tool"}),
        )
    )
    assert step.action_type == "tool"
    assert step.action_content == "calculator"
    assert step.is_terminal is True
