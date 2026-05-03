"""Unit tests for src.agent_loop."""

import asyncio

import pytest

from src.agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    SearchEvaluationConfig,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchResultEvaluator,
    SearchContext,
    SearchResult,
    SingleTurnAgentLoop,
    get_registered_agent_loop,
    list_registered_agent_loops,
    register,
)


class DummyTokenizerWithTemplate:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [11, 12, 13, 14, 15]


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
    prompt_ids = asyncio.run(loop.build_prompt_ids([{"role": "user", "content": "hello"}]))
    assert prompt_ids == [13, 14, 15]


def test_build_prompt_ids_falls_back_to_encode():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=4, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids([{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}])
    )
    assert len(prompt_ids) == 4


def test_generate_response_ids_truncates_to_response_length():
    server_manager = DummyServerManager([1, 2, 3, 4, 5])
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=10, response_length=3),
    )
    response_ids = asyncio.run(loop.generate_response_ids([9, 9], {"temperature": 0.1}, request_id="req-1"))
    assert response_ids == [1, 2, 3]
    assert server_manager.calls[0]["request_id"] == "req-1"


def test_single_turn_agent_loop_returns_expected_output():
    server_manager = DummyServerManager([21, 22, 23, 24])
    loop = SingleTurnAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=4, response_length=2),
    )
    output = asyncio.run(loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.7}))
    assert output.prompt_ids == [12, 13, 14, 15]
    assert output.response_ids == [21, 22]
    assert output.response_mask == [1, 1]
    assert output.num_turns == 1
    assert "generate_sequences" in output.metrics
    assert output.request_id is not None


def test_register_stores_class_by_name():
    @register("test_agent_loop")
    class RegisteredLoop(ConcreteAgentLoop):
        pass

    assert RegisteredLoop.__name__ == "RegisteredLoop"


def test_get_registered_agent_loop_returns_single_turn_loop():
    registered = get_registered_agent_loop("single_turn_agent")
    assert registered is SingleTurnAgentLoop


def test_list_registered_agent_loops_includes_single_turn():
    assert "single_turn_agent" in list_registered_agent_loops()


def test_get_registered_agent_loop_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown agent loop"):
        get_registered_agent_loop("missing_loop")


def test_generate_response_ids_supports_sync_server_manager():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummySyncServerManager([7, 8, 9, 10]),
        config=AgentLoopConfig(prompt_length=10, response_length=2),
    )
    response_ids = asyncio.run(loop.generate_response_ids([1, 2], {"temperature": 0.3}, request_id="req-sync"))
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
    evaluation = evaluator.evaluate_round([
        SearchContext(
            query="alpha",
            results=[SearchResult(contents='"Alpha"\nbody', score=0.4)],
        )
    ])

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
    evaluation = evaluator.evaluate_round([
        SearchContext(
            query="alpha",
            results=[
                SearchResult(contents='"Alpha"\nbody', score=0.9),
                SearchResult(contents='"Alpha 2"\nbody', score=0.8),
            ],
        )
    ])

    assert evaluation.is_sufficient is True
    assert "Verdict: SUFFICIENT" in evaluation.to_feedback_block()


class FakeSearchClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.fetch_calls = []
        self.fetch_responses = {}

    async def retrieve(self, queries, topk=None):
        del topk
        self.calls.append(list(queries))
        return self.responses[tuple(queries)]

    async def fetch_urls(self, urls):
        self.fetch_calls.append(list(urls))
        return self.fetch_responses[tuple(urls)]


def test_search_agent_loop_supports_plan_parallel_search_and_research_rounds():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<plan>Compare two sources and validate with a follow-up search.</plan>"),
        tokenizer.encode("<searches>\n- first query\n- second query\n</searches>"),
        tokenizer.encode("<searches><query>refined query</query></searches>"),
        tokenizer.encode("<answer>Final report [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
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

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

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

    asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    second_prompt = "".join(chr(token) for token in loop.server_manager.calls[1]["prompt_ids"])
    assert "<search_evaluation>" in second_prompt
    assert "Verdict: INSUFFICIENT" in second_prompt
    assert "keep searching" in second_prompt


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
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    second_prompt = "".join(chr(token) for token in loop.server_manager.calls[1]["prompt_ids"])
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

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    third_prompt = "".join(chr(token) for token in loop.server_manager.calls[2]["prompt_ids"])
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

    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0}))

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
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', url="https://example.com/a")],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(contents="Full page body", title="Doc A", url="https://example.com/a"),
        ],
    }
    loop._search_client = fake_client

    asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    third_prompt = "".join(chr(token) for token in loop.server_manager.calls[2]["prompt_ids"])
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
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha",): [
                [SearchResult(contents='"Doc A"\nAlpha body', url="https://example.com/a")],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(contents="Full page body", title="Doc A", url="https://example.com/a"),
        ],
    }
    loop._search_client = fake_client

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    assert fake_client.calls == [["alpha"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    assert output.metrics["search_rounds"] == 1.0
    assert output.metrics["search_queries"] == 1.0
    assert output.metrics["fetched_pages"] == 1.0


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
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=2),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice", "Lara Croft game developer"): [
                [SearchResult(contents='"Voice"\nAlice David', url="https://example.com/voice")],
                [SearchResult(contents='"Developer"\nCrystal Dynamics', url="https://example.com/dev")],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    assert output.context.tasks == {
        "T1": "identify the voice actor",
        "T2": "identify the developer",
    }
    assert output.context.turns[0].task_id == "T1"
    assert output.context.turns[1].task_id == "T2"
    assert output.metrics["active_subquestions"] == 2.0


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
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice",): [
                [SearchResult(contents='"Voice"\nAlice David', url="https://example.com/voice")],
            ],
            ("Lara Croft game developer",): [
                [SearchResult(contents='"Developer"\nCrystal Dynamics', url="https://example.com/dev")],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0}))

    third_prompt = "".join(chr(token) for token in loop.server_manager.calls[2]["prompt_ids"])
    assert "T2: identify the developer" in third_prompt
    assert "<answer_feedback>" in third_prompt
    assert output.context.num_rounds == 2


def test_search_client_config_derives_fetch_url_from_retrieve_url():
    from src.agent_loop.search_client import SearchClientConfig

    cases = [
        ("http://localhost:8000/retrieve", "http://localhost:8000/fetch"),
        ("http://localhost:8000/retrieve/", "http://localhost:8000/fetch"),
        ("http://host:9000/api/retrieve", "http://host:9000/api/fetch"),
        ("http://host/other", "http://host/other/fetch"),
    ]
    for url, expected in cases:
        assert SearchClientConfig(url=url).get_fetch_url() == expected, f"Failed for {url!r}"


def test_search_agent_loop_processes_search_and_fetch_in_same_turn():
    """When the model emits <searches> and <fetch> in the same turn, both are
    executed and their results appear in a single observation message."""
    tokenizer = DummyTokenizerWithEncode()
    combined = "<searches>\nalpha query\n</searches><fetch>https://example.com/a</fetch>"
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(min_results_per_query=1, min_total_results=1),
        ),
    )
    fake_client = FakeSearchClient(
        {("alpha query",): [[SearchResult(contents='"Doc A"\nAlpha body', url="https://example.com/a")]]}
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(contents="Full page body", title="Doc A", url="https://example.com/a")
        ]
    }
    loop._search_client = fake_client

    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0}))

    # Both search and fetch fired in turn 0.
    assert fake_client.calls == [["alpha query"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    # Both observations are in the same injected user message (turn 1 prompt).
    second_prompt = "".join(chr(t) for t in loop.server_manager.calls[1]["prompt_ids"])
    assert "<information>" in second_prompt
    assert "<full_page>" in second_prompt
    assert output.num_turns == 2
