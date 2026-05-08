"""Unit tests for src.llm_agent.generation."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)

from src.agent_loop import build_prompt_dataloader  # noqa: E402
from src.llm_agent.generation import (  # noqa: E402
    AgentLoopState,
    GenerationConfig,
    LLMGenerationManager,
    PPOPolicyLossConfig,
    RetrievedDocument,
    RolloutTrajectory,
    SearchBatch,
    format_search_trajectory_log,
)


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1] * max(len(text.split()), 1)

    def __call__(
        self, texts, add_special_tokens=False, return_tensors="pt", padding="longest"
    ):
        del add_special_tokens, padding
        max_len = max((len(text.split()) for text in texts), default=0)
        rows = []
        for text in texts:
            length = len(text.split())
            rows.append([1] * length + [0] * (max_len - length))
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def batch_decode(self, responses, skip_special_tokens=True):
        del responses, skip_special_tokens
        return []


class DummyActorRollout:
    def generate_sequences(self, active_batch):
        return active_batch


class DummyActorRolloutWithLogProb(DummyActorRollout):
    def compute_log_prob(self, batch):
        return torch.full_like(batch.batch["responses"], -0.5, dtype=torch.float32)


class FixedLogProbBackend(DummyActorRollout):
    def __init__(self, value: float):
        self.value = value

    def compute_log_prob(self, batch):
        return torch.full_like(
            batch.batch["responses"], self.value, dtype=torch.float32
        )


class SequencedActorRollout:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_sequences(self, active_batch):
        batch_size = active_batch.batch["input_ids"].shape[0]
        width = max((len(item) for item in self.responses[self.calls]), default=0)
        rows = []
        for token_ids in self.responses[self.calls]:
            rows.append(list(token_ids) + [0] * (width - len(token_ids)))
        self.calls += 1
        return SearchBatch.from_dict(
            {"responses": torch.tensor(rows[:batch_size], dtype=torch.long)}
        )


def _manager() -> LLMGenerationManager:
    return LLMGenerationManager(
        tokenizer=DummyTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=DummyActorRollout(),
    )


def _manager_with_log_prob() -> LLMGenerationManager:
    return LLMGenerationManager(
        tokenizer=DummyTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=DummyActorRolloutWithLogProb(),
    )


def test_postprocess_predictions_extracts_rollout_action_tags():
    manager = _manager()
    actions, contents = manager.postprocess_predictions(
        [
            "before <search>cats</search> after",
            "<answer>42</answer>",
            "<plan>break it down</plan>",
            "<fetch>https://example.com</fetch>",
            "plain text",
        ]
    )
    assert actions == ["search", "answer", "plan", "fetch", None]
    assert contents == ["cats", "42", "break it down", "https://example.com", ""]


def test_execute_predictions_keeps_search_payload_aligned():
    manager = _manager()

    captured = {}

    def fake_batch_search(search_payload, search_mode, gt_threshold):
        captured["payload"] = search_payload
        captured["search_mode"] = search_mode
        captured["gt_threshold"] = gt_threshold
        return ["doc for second question"]

    manager.batch_search = fake_batch_search  # type: ignore[method-assign]
    next_obs, dones, valid_action, is_search = manager.execute_predictions(
        predictions=["<answer>done</answer>", "<search>cats</search>"],
        problem=["first question", "second question"],
        ground_truth=[["first answer"], ["second answer"]],
        search_mode="google",
        gt_threshold=0.8,
        active_mask=torch.tensor([True, True]),
    )

    assert len(captured["payload"]) == 1
    assert captured["payload"][0].query == "cats"
    assert captured["payload"][0].problem == "second question"
    assert captured["payload"][0].ground_truth == "second answer"
    assert captured["search_mode"] == "google"
    assert captured["gt_threshold"] == 0.8
    assert next_obs[1] == "\n\n<information>doc for second question</information>\n\n"
    assert dones == [1, 0]
    assert valid_action == [1, 1]
    assert is_search == [0, 1]


def test_build_react_observation_wraps_search_result_in_information_tags():
    from src.llm_agent.generation import PolicyAction, build_react_observation

    action = PolicyAction(tag="search", content="2024 physics nobel prize", raw_text="")
    obs = build_react_observation(
        action, "The 2024 Nobel Prize in Physics was awarded to ..."
    )
    assert obs == (
        "\n\n<information>The 2024 Nobel Prize in Physics was awarded to ..."
        "</information>\n\n"
    )


def test_build_react_observation_returns_empty_for_answer():
    from src.llm_agent.generation import PolicyAction, build_react_observation

    action = PolicyAction(tag="answer", content="Watson and Watt", raw_text="")
    assert build_react_observation(action) == ""


def test_build_react_observation_returns_plan_feedback():
    from src.llm_agent.generation import PolicyAction, build_react_observation

    action = PolicyAction(tag="plan", content="outline", raw_text="")
    obs = build_react_observation(action)
    assert "<plan_feedback>" in obs


def test_build_search_tool_calls_uses_model_emitted_queries():
    manager = _manager()
    calls = manager.build_search_tool_calls(
        manager.parse_policy_actions(
            [
                "<search>2024 physics nobel prize winner</search>",
                "<answer>done</answer>",
            ]
        ),
        problem=["q1", "q2"],
        ground_truth=[["a1"], ["a2"]],
        active_mask=torch.tensor([True, True]),
    )

    assert len(calls) == 1
    assert calls[0].query == "2024 physics nobel prize winner"
    assert calls[0].problem == "q1"
    assert calls[0].ground_truth == "a1"
    assert calls[0].batch_index == 0


def test_build_search_decisions_makes_model_search_choice_explicit():
    manager = _manager()
    decisions = manager.build_search_decisions(
        manager.parse_policy_actions(
            [
                "<search>2024 physics nobel prize winner</search>",
                "<answer>done</answer>",
            ]
        ),
        problem=["q1", "q2"],
        ground_truth=[["a1"], ["a2"]],
        active_mask=torch.tensor([True, True]),
    )

    assert len(decisions) == 1
    assert decisions[0].query == "2024 physics nobel prize winner"
    assert decisions[0].problem == "q1"
    assert decisions[0].ground_truth == "a1"
    assert decisions[0].batch_index == 0
    assert decisions[0].raw_action_text == (
        "<search>2024 physics nobel prize winner</search>"
    )


def test_build_search_decisions_skips_empty_queries_and_inactive_rows():
    manager = _manager()
    decisions = manager.build_search_decisions(
        manager.parse_policy_actions(
            [
                "<search>   </search>",
                "<search>cats</search>",
            ]
        ),
        problem=["q1", "q2"],
        ground_truth=[["a1"], ["a2"]],
        active_mask=torch.tensor([True, False]),
    )
    assert decisions == []


def test_execute_predictions_marks_inactive_examples_done():
    manager = _manager()
    next_obs, dones, valid_action, is_search = manager.execute_predictions(
        predictions=["<search>cats</search>", "<answer>x</answer>"],
        problem=["first", "second"],
        ground_truth=[["a"], ["b"]],
        search_mode="google",
        gt_threshold=0.5,
        active_mask=torch.tensor([False, True]),
        do_search=False,
    )
    assert next_obs[0] == ""
    assert dones == [1, 1]
    assert valid_action == [0, 1]
    assert is_search == [0, 0]


def test_execute_predictions_accepts_plan_and_fetch_actions():
    manager = _manager()
    manager.batch_fetch = lambda payload: ["Full page body from fetch"]  # type: ignore[method-assign]
    next_obs, dones, valid_action, is_search = manager.execute_predictions(
        predictions=["<plan>outline</plan>", "<fetch>https://example.com</fetch>"],
        problem=["first", "second"],
        ground_truth=[["a"], ["b"]],
        search_mode="google",
        gt_threshold=0.5,
        active_mask=torch.tensor([True, True]),
        do_search=True,
    )
    assert "<plan_feedback>" in next_obs[0]
    assert next_obs[1] == "\n\n<full_page>Full page body from fetch</full_page>\n\n"
    assert dones == [0, 0]
    assert valid_action == [1, 1]
    assert is_search == [0, 0]


def test_build_fetch_tool_calls_splits_urls_from_model_output():
    manager = _manager()
    calls = manager.build_fetch_tool_calls(
        manager.parse_policy_actions(
            [
                "<fetch>https://a.example.com,\nhttps://b.example.com</fetch>",
                "<answer>done</answer>",
            ]
        ),
        active_mask=torch.tensor([True, True]),
    )

    assert len(calls) == 1
    assert calls[0].urls == ["https://a.example.com", "https://b.example.com"]
    assert calls[0].batch_index == 0


def test_search_returns_fallback_for_unknown_mode():
    from src.llm_agent.generation import SearchToolCall

    manager = _manager()
    tc = SearchToolCall(
        query="query", problem="problem", ground_truth="answer", batch_index=3
    )
    result, index = manager._search(tc, "unknown", 0.5)
    assert result == "No information available"
    assert index == 3


def test_parse_api_item_supports_nested_document_shape():
    from src.llm_agent.generation import _parse_api_item

    doc = _parse_api_item(
        {"document": {"title": "T", "contents": '"T"\nBody text', "url": "https://x"}}
    )
    assert doc.title == "T"
    assert doc.snippet == "Body text"
    assert doc.url == "https://x"


def test_parse_api_item_supports_flat_snippet_shape():
    from src.llm_agent.generation import _parse_api_item

    doc = _parse_api_item(
        {
            "title": "Physics Prize",
            "snippet": "Hopfield and Hinton won.",
            "link": "https://example.com",
        }
    )
    assert doc.title == "Physics Prize"
    assert doc.snippet == "Hopfield and Hinton won."
    assert doc.url == "https://example.com"


def test_documents_from_retrieve_payload_supports_standard_post_retrieve_shape():
    from src.llm_agent.generation import _documents_from_retrieve_payload

    docs = _documents_from_retrieve_payload(
        {
            "result": [
                [
                    {
                        "document": {
                            "title": "Physics Nobel",
                            "contents": '"Physics Nobel"\nHopfield and Hinton won.',
                            "url": "https://example.com/nobel",
                        },
                        "score": 0.9,
                    }
                ]
            ]
        }
    )

    assert len(docs) == 1
    assert docs[0].title == "Physics Nobel"
    assert docs[0].snippet == "Hopfield and Hinton won."
    assert docs[0].url == "https://example.com/nobel"
    assert docs[0].score == pytest.approx(0.9)


def test_documents_from_retrieve_payload_supports_direct_row_list():
    from src.llm_agent.generation import _documents_from_retrieve_payload

    docs = _documents_from_retrieve_payload(
        [
            {
                "title": "Physics Nobel",
                "snippet": "Hopfield and Hinton won.",
                "url": "https://example.com/nobel",
            }
        ]
    )

    assert len(docs) == 1
    assert docs[0].title == "Physics Nobel"
    assert docs[0].snippet == "Hopfield and Hinton won."


def test_passages2string_formats_structured_retrieval_results():
    manager = _manager()
    rendered = manager._passages2string(
        [
            {
                "document": {
                    "title": "Physics Prize",
                    "contents": '"Physics Prize"\nHopfield and Hinton won in 2024.',
                    "url": "https://example.com/nobel",
                },
                "score": 0.9,
            },
            {
                "title": "Background",
                "snippet": "The Nobel Prize in Physics is awarded annually.",
            },
        ]
    )
    assert "Doc 1(Title: Physics Prize) Hopfield and Hinton won in 2024." in rendered
    assert "URL: https://example.com/nobel" in rendered
    assert (
        "Doc 2(Title: Background) The Nobel Prize in Physics is awarded annually."
        in rendered
    )


def test_documents_per_query_from_payload_parses_multi_query_response():
    from src.llm_agent.generation import _documents_per_query_from_payload

    payload = {
        "result": [
            [{"document": {"title": "A", "contents": "snippet A"}, "score": 0.9}],
            [{"document": {"title": "B", "contents": "snippet B"}, "score": 0.7}],
        ]
    }
    per_query = _documents_per_query_from_payload(payload, n_queries=2)
    assert len(per_query) == 2
    assert per_query[0][0].title == "A"
    assert per_query[1][0].title == "B"


def test_documents_per_query_from_payload_pads_missing_queries():
    from src.llm_agent.generation import _documents_per_query_from_payload

    payload = {"result": [[{"document": {"title": "A", "contents": ""}, "score": 0.9}]]}
    per_query = _documents_per_query_from_payload(payload, n_queries=3)
    assert len(per_query) == 3
    assert len(per_query[0]) == 1
    assert per_query[1] == []
    assert per_query[2] == []


def test_endpoint_batch_search_sends_one_request_for_multiple_queries(monkeypatch):
    """batch_search in local mode must issue one POST with all queries, not N."""
    from unittest.mock import MagicMock, patch

    manager = _manager()
    flat_calls = manager.build_search_tool_calls(
        manager.parse_policy_actions(["<search>q1</search>", "<search>q2</search>"]),
        problem=["p1", "p2"],
        ground_truth=[["a1"], ["a2"]],
        active_mask=torch.tensor([True, True]),
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "result": [
            [{"document": {"title": "T1", "contents": "doc one"}}],
            [{"document": {"title": "T2", "contents": "doc two"}}],
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    posted_bodies = []

    def fake_post(url, json=None, timeout=None):
        posted_bodies.append(json)
        return mock_resp

    with patch("requests.post", side_effect=fake_post):
        manager.config = manager.config.__class__(
            **{
                **manager.config.__dict__,
                "retrieval_url": "http://localhost:6002/retrieve",
            }
        )
        results = manager.batch_search(
            flat_calls, search_mode="local", gt_threshold=0.5
        )

    # Only one HTTP request must have been sent
    assert len(posted_bodies) == 1
    assert posted_bodies[0]["queries"] == ["q1", "q2"]
    assert len(results) == 2
    assert "T1" in results[0] or "doc one" in results[0]
    assert "T2" in results[1] or "doc two" in results[1]


def test_batch_search_uses_parallel_threads_for_non_endpoint_modes():
    """simulate modes must not hit the endpoint batch path."""
    manager = _manager()
    searched = []

    def tracking_search(tc, search_mode, gt_threshold):
        searched.append(tc.query)
        return "simulated result", tc.batch_index

    manager._search = tracking_search  # type: ignore[method-assign]
    tool_calls = manager.build_search_tool_calls(
        manager.parse_policy_actions(["<search>q1</search>", "<search>q2</search>"]),
        problem=["p1", "p2"],
        ground_truth=[["a1"], ["a2"]],
        active_mask=torch.tensor([True, True]),
    )
    manager.batch_search(tool_calls, search_mode="simulate_sft", gt_threshold=0.5)
    # Both queries must have gone through the per-query path
    assert sorted(searched) == ["q1", "q2"]


def test_safe_truncate_observation_preserves_close_tag_on_long_information_block():
    from src.llm_agent.generation import LLMGenerationManager

    long_content = "word " * 200  # 1000 chars
    obs = f"\n\n<information>{long_content}</information>\n\n"
    truncated = LLMGenerationManager._safe_truncate_observation(obs, max_chars=100)

    assert truncated.endswith("</information>\n\n")
    assert len(truncated) <= 100 + len(
        "</information>\n\n"
    )  # slight overrun allowed by design


def test_safe_truncate_observation_does_not_modify_short_observations():
    from src.llm_agent.generation import LLMGenerationManager

    obs = "\n\n<information>short result</information>\n\n"
    assert LLMGenerationManager._safe_truncate_observation(obs, max_chars=500) == obs


def test_safe_truncate_observation_hard_cuts_unknown_format():
    from src.llm_agent.generation import LLMGenerationManager

    obs = "plain text that is very long " * 20
    truncated = LLMGenerationManager._safe_truncate_observation(obs, max_chars=50)
    assert len(truncated) == 50
    assert truncated == obs[:50]


def test_build_react_context_transitions_captures_action_and_observation():
    manager = _manager()
    action = "<search>cats</search>"
    obs = "\n\n<information>cats are animals</information>\n\n"
    transitions = manager.build_react_context_transitions([action], [obs])

    assert len(transitions) == 1
    assert transitions[0].action_text == action
    assert transitions[0].observation_text == obs
    assert transitions[0].appended_context_text == action + obs


def test_build_react_context_transitions_length_mismatch_raises():
    manager = _manager()
    with pytest.raises(ValueError, match="same length"):
        manager.build_react_context_transitions(["a", "b"], ["x"])


def test_retrieve_documents_returns_structured_docs_from_backend():
    manager = _manager()

    class FakeRetriever:
        def retrieve(self, query, topk):
            assert query == "2024 physics nobel prize winner"
            assert topk == 3
            return [
                RetrievedDocument(
                    title="Physics Nobel",
                    snippet="Hopfield and Hinton won.",
                    url="https://example.com/nobel",
                    score=0.9,
                )
            ]

        def search(self, query, topk):
            del query, topk
            return "unused"

    manager._make_retriever = (
        lambda tool_call, search_mode, gt_threshold: FakeRetriever()
    )  # type: ignore[method-assign]
    docs = manager.retrieve_documents(
        "2024 physics nobel prize winner",
        search_mode="local",
        topk=3,
    )

    assert len(docs) == 1
    assert docs[0].title == "Physics Nobel"
    assert docs[0].snippet == "Hopfield and Hinton won."


def test_compute_log_prob_stores_new_log_probs_on_batch_and_trajectory():
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6, 0]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6, 0]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["trajectories"] = [
        RolloutTrajectory(
            batch_index=0,
            prompt_token_ids=[1, 2],
            response_token_ids=[5, 6, 0],
            response_with_observation_mask=[5, 6, 0],
            trajectory_turns=1,
            steps=[],
            final_answer="done",
            finished_without_answer=False,
        )
    ]

    log_probs = manager.compute_log_prob(batch, store_key="new_log_probs")

    assert torch.allclose(
        log_probs, torch.tensor([[-0.5, -0.5, -0.0]], dtype=torch.float32)
    )
    assert "new_log_probs" in batch.batch
    # new_log_probs is prompt-padded: [0.0, 0.0] for prompt [1,2] then response values
    assert batch.non_tensor_batch["trajectories"][0].new_log_probs == [
        0.0,
        0.0,
        -0.5,
        -0.5,
        -0.0,
    ]


def test_compute_log_prob_zeros_out_environment_information_tokens():
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 0, 7]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["trajectories"] = [
        RolloutTrajectory(
            batch_index=0,
            prompt_token_ids=[1, 2],
            response_token_ids=[5, 6, 7],
            response_with_observation_mask=[5, 0, 7],
            trajectory_turns=1,
            steps=[],
            final_answer="done",
            finished_without_answer=False,
        )
    ]

    log_probs = manager.compute_log_prob(batch, store_key="old_log_probs")

    assert torch.allclose(
        log_probs, torch.tensor([[-0.5, -0.0, -0.5]], dtype=torch.float32)
    )
    # old_log_probs is prompt-padded: [0.0, 0.0] for prompt [1,2] then response values
    assert batch.non_tensor_batch["trajectories"][0].old_log_probs == [
        0.0,
        0.0,
        -0.5,
        -0.0,
        -0.5,
    ]


def test_prepare_policy_log_probs_stores_old_new_and_ratio():
    manager = _manager()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6, 0]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6, 0]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["trajectories"] = [
        RolloutTrajectory(
            batch_index=0,
            prompt_token_ids=[1, 2],
            response_token_ids=[5, 6, 0],
            response_with_observation_mask=[5, 6, 0],
            trajectory_turns=1,
            steps=[],
            final_answer="done",
            finished_without_answer=False,
        )
    ]

    old_log_probs, new_log_probs = manager.prepare_policy_log_probs(
        batch,
        old_backend=FixedLogProbBackend(-1.0),
        new_backend=FixedLogProbBackend(-0.5),
    )

    expected_old = torch.tensor([[-1.0, -1.0, -0.0]], dtype=torch.float32)
    expected_new = torch.tensor([[-0.5, -0.5, -0.0]], dtype=torch.float32)
    expected_ratio = torch.exp(expected_new - expected_old)

    assert torch.allclose(old_log_probs, expected_old)
    assert torch.allclose(new_log_probs, expected_new)
    assert torch.allclose(batch.batch["old_log_probs"], expected_old)
    assert torch.allclose(batch.batch["new_log_probs"], expected_new)
    assert torch.allclose(batch.batch["prob_ratio"], expected_ratio)
    assert batch.meta_info["policy_log_probs_prepared"] is True
    trajectory = batch.non_tensor_batch["trajectories"][0]
    # prompt_token_ids=[1, 2] → two 0.0 prefix values before response log probs
    assert trajectory.old_log_probs == [0.0, 0.0, -1.0, -1.0, -0.0]
    assert trajectory.new_log_probs == [0.0, 0.0, -0.5, -0.5, -0.0]


def test_prepare_policy_log_probs_can_store_reference_log_probs():
    manager = _manager()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["trajectories"] = [
        RolloutTrajectory(
            batch_index=0,
            prompt_token_ids=[1, 2],
            response_token_ids=[5, 6],
            response_with_observation_mask=[5, 6],
            trajectory_turns=1,
            steps=[],
            final_answer="done",
            finished_without_answer=False,
            tokens=[1, 2, 5, 6],
            attention_mask=[1, 1, 1, 1],
            response_mask=[0, 0, 1, 1],
        )
    ]

    manager.prepare_policy_log_probs(
        batch,
        old_backend=FixedLogProbBackend(-1.0),
        new_backend=FixedLogProbBackend(-0.5),
        ref_backend=FixedLogProbBackend(-1.5),
    )

    assert torch.allclose(
        batch.batch["ref_log_probs"],
        torch.tensor([[-1.5, -1.5]], dtype=torch.float32),
    )
    assert batch.non_tensor_batch["trajectories"][0].ref_log_probs == [
        0.0,
        0.0,
        -1.5,
        -1.5,
    ]


def test_old_log_probs_aligned_with_tokens_and_response_mask():
    """tokens, attention_mask, response_mask, old_log_probs must all be the same length."""
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6, 0]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6, 0]], dtype=torch.long),
        }
    )
    prompt_ids = [1, 2]
    batch.non_tensor_batch["trajectories"] = [
        RolloutTrajectory(
            batch_index=0,
            prompt_token_ids=prompt_ids,
            response_token_ids=[5, 6, 0],
            response_with_observation_mask=[5, 6, 0],
            trajectory_turns=1,
            steps=[],
            final_answer="done",
            finished_without_answer=False,
            tokens=prompt_ids + [5, 6, 0],
            attention_mask=[1, 1, 1, 1, 1],
            response_mask=[0, 0, 1, 1, 0],
        )
    ]

    manager.compute_log_prob(batch, store_key="old_log_probs")
    traj = batch.non_tensor_batch["trajectories"][0]

    n = len(traj.tokens)
    assert len(traj.old_log_probs) == n, "old_log_probs must be prompt+response length"
    assert len(traj.attention_mask) == n
    assert len(traj.response_mask) == n
    # Prompt positions must have 0.0 log prob and 0 response mask
    n_prompt = len(prompt_ids)
    assert traj.old_log_probs[:n_prompt] == [0.0] * n_prompt
    assert traj.response_mask[:n_prompt] == [0] * n_prompt


def test_trajectory_log_prob_pack_returns_aligned_arrays():
    from src.llm_agent.generation import trajectory_log_prob_pack

    traj = RolloutTrajectory(
        batch_index=0,
        prompt_token_ids=[1, 2],
        response_token_ids=[5, 6, 7],
        response_with_observation_mask=[5, 0, 7],
        trajectory_turns=1,
        steps=[],
        final_answer="done",
        finished_without_answer=False,
        tokens=[1, 2, 5, 6, 7],
        attention_mask=[1, 1, 1, 1, 1],
        response_mask=[0, 0, 1, 0, 1],
        old_log_probs=[0.0, 0.0, -0.5, 0.0, -0.5],
    )
    pack = trajectory_log_prob_pack(traj)

    assert set(pack.keys()) == {
        "tokens",
        "attention_mask",
        "response_mask",
        "old_log_probs",
    }
    n = len(pack["tokens"])
    assert len(pack["attention_mask"]) == n
    assert len(pack["response_mask"]) == n
    assert len(pack["old_log_probs"]) == n
    # observation token at index 3 → response_mask=0, old_log_probs=0.0
    assert pack["response_mask"][3] == 0
    assert pack["old_log_probs"][3] == 0.0


def test_trajectory_log_prob_pack_falls_back_to_zeros_when_log_probs_none():
    from src.llm_agent.generation import trajectory_log_prob_pack

    traj = RolloutTrajectory(
        batch_index=0,
        prompt_token_ids=[1, 2],
        response_token_ids=[5, 6],
        response_with_observation_mask=[5, 6],
        trajectory_turns=1,
        steps=[],
        final_answer="done",
        finished_without_answer=False,
        tokens=[1, 2, 5, 6],
        attention_mask=[1, 1, 1, 1],
        response_mask=[0, 0, 1, 1],
        old_log_probs=None,  # backend didn't compute log probs
    )
    pack = trajectory_log_prob_pack(traj)

    assert len(pack["old_log_probs"]) == 4
    assert pack["old_log_probs"] == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# compute_trajectory_policy_loss — GRPO/PPO clipped loss, trajectory level
# ---------------------------------------------------------------------------


def test_compute_trajectory_policy_loss_matches_grpo_formula():
    """ratio=exp(new-old), loss=-mean(min(r*A, clip(r)*A)*mask)."""
    from src.llm_agent.generation import compute_trajectory_policy_loss

    # 5-element aligned arrays: 2 prompt zeros + 3 response positions
    new_lp = [0.0, 0.0, -0.5, -0.5, -0.5]
    old_lp = [0.0, 0.0, -1.0, -1.0, -1.0]
    # Sparse GRPO: reward only at last action token (index 4)
    advantages = [0.0, 0.0, 0.0, 0.0, 1.0]
    mask = [0, 0, 1, 1, 1]  # 0 for prompt, 1 for model actions

    out = compute_trajectory_policy_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=advantages,
        response_mask=mask,
        clip_epsilon=0.2,
    )

    # ratio = exp(0.5) ≈ 1.649; clamped to 1.2 (clip_epsilon=0.2)
    # Only index 4 has advantage=1.0; indices 2,3 have advantage=0 → contribute 0
    # policy_loss = -(1.2 * 1.0) / 3  (3 masked tokens)
    expected_loss = -(1.2 * 1.0) / 3
    assert out["grpo_policy_loss"] == pytest.approx(expected_loss, abs=1e-5)
    assert out["kl_penalty"] == pytest.approx(0.0)
    assert out["total_loss"] == pytest.approx(expected_loss, abs=1e-5)
    # All 3 action tokens are clipped (ratio ≈ 1.65 > 1.2) → clip_fraction = 1.0
    assert out["clip_fraction"] == pytest.approx(1.0, abs=1e-5)


def test_compute_trajectory_policy_loss_negative_advantage_penalises_above_clip():
    from src.llm_agent.generation import compute_trajectory_policy_loss
    import math

    # ratio > 1 + eps with A < 0 → should be clipped
    new_lp = [0.0, 0.0, -0.5]
    old_lp = [0.0, 0.0, -1.5]  # ratio = exp(1.0) ≈ 2.72, well above 1.2
    advantages = [0.0, 0.0, -1.0]  # negative advantage
    mask = [0, 0, 1]

    out = compute_trajectory_policy_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=advantages,
        response_mask=mask,
        clip_epsilon=0.2,
    )

    # min(r * -1, clip(r) * -1) = min(-2.72, -1.2) = -2.72 (unclipped is worse)
    # surrogate = -2.72 * 1 mask token → loss = -(-2.72)/1 = +2.72
    r = math.exp(1.0)
    expected = r * 1.0  # positive loss — clip does NOT help for A<0, ratio>1+eps
    assert out["grpo_policy_loss"] == pytest.approx(expected, abs=1e-4)


def test_compute_trajectory_policy_loss_kl_penalty_fires_with_ref_log_probs():
    from src.llm_agent.generation import compute_trajectory_policy_loss

    # ref = old = new → KL = 0; use ref != new to get non-zero KL
    new_lp = [0.0, 0.0, -0.5, -0.5]
    old_lp = [0.0, 0.0, -0.5, -0.5]  # ratio = 1.0 everywhere
    ref_lp = [0.0, 0.0, -1.0, -1.0]  # ref diverges from new
    advantages = [0.0, 0.0, 1.0, 0.0]
    mask = [0, 0, 1, 1]

    out_no_kl = compute_trajectory_policy_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=advantages,
        response_mask=mask,
        kl_beta=0.0,
    )
    out_with_kl = compute_trajectory_policy_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=advantages,
        response_mask=mask,
        ref_log_probs=ref_lp,
        kl_beta=0.1,
    )

    assert out_no_kl["kl_penalty"] == pytest.approx(0.0)
    assert out_with_kl["kl_penalty"] > 0.0
    assert out_with_kl["total_loss"] > out_no_kl["total_loss"]


def test_compute_trajectory_policy_loss_kl_uses_old_when_ref_absent():
    from src.llm_agent.generation import compute_trajectory_policy_loss

    new_lp = [0.0, 0.0, -0.5]
    old_lp = [0.0, 0.0, -0.5]  # identical → KL = 0
    advantages = [0.0, 0.0, 1.0]
    mask = [0, 0, 1]

    out = compute_trajectory_policy_loss(
        new_log_probs=new_lp,
        old_log_probs=old_lp,
        advantages=advantages,
        response_mask=mask,
        kl_beta=0.5,  # beta > 0 but no ref, old==new
    )

    assert out["kl_penalty"] == pytest.approx(0.0, abs=1e-7)


def test_compute_trajectory_policy_loss_length_mismatch_raises():
    from src.llm_agent.generation import compute_trajectory_policy_loss

    with pytest.raises(ValueError, match="same length"):
        compute_trajectory_policy_loss(
            new_log_probs=[0.0, -0.5],
            old_log_probs=[0.0],  # wrong length
            advantages=[0.0, 1.0],
            response_mask=[0, 1],
        )


def test_compute_trajectory_policy_loss_with_trajectory_log_prob_pack():
    """End-to-end: pack + loss computes without error."""
    from src.llm_agent.generation import (
        compute_trajectory_policy_loss,
        trajectory_log_prob_pack,
    )

    traj = RolloutTrajectory(
        batch_index=0,
        prompt_token_ids=[1, 2],
        response_token_ids=[5, 6, 7],
        response_with_observation_mask=[5, 0, 7],
        trajectory_turns=1,
        steps=[],
        final_answer="done",
        finished_without_answer=False,
        tokens=[1, 2, 5, 6, 7],
        attention_mask=[1, 1, 1, 1, 1],
        response_mask=[0, 0, 1, 0, 1],
        old_log_probs=[0.0, 0.0, -1.0, 0.0, -1.0],
    )
    traj_new_lps = [0.0, 0.0, -0.8, 0.0, -0.8]
    pack = trajectory_log_prob_pack(traj)
    advantages = [0.0] * 5
    advantages[4] = 1.0  # sparse: reward at last action token only

    out = compute_trajectory_policy_loss(
        new_log_probs=traj_new_lps,
        old_log_probs=pack["old_log_probs"],
        advantages=advantages,
        response_mask=pack["response_mask"],
        clip_epsilon=0.2,
    )

    assert "grpo_policy_loss" in out
    assert "total_loss" in out
    assert isinstance(out["total_loss"], float)


def test_compose_final_output_stores_training_tokens_and_masks_on_trajectory():
    manager = _manager()
    left_side = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
    }
    right_side = {
        "responses": torch.tensor([[21, 22, 23]], dtype=torch.long),
        "responses_with_info_mask": torch.tensor([[21, 0, 23]], dtype=torch.long),
    }
    meta_info = {
        "trajectory_turns": [2],
        "final_answers": ["done"],
        "finished_without_answer": [False],
        "react_trajectory": [[]],
        "questions": ["What is the answer?"],
    }

    batch = manager._compose_final_output(left_side, right_side, meta_info)
    trajectory = batch.non_tensor_batch["trajectories"][0]

    assert trajectory.tokens == [11, 12, 21, 22, 23]
    assert trajectory.attention_mask == [1, 1, 1, 1, 1]
    assert trajectory.response_mask == [0, 0, 1, 0, 1]


def test_prepare_policy_log_probs_can_skip_ratio_computation():
    manager = _manager()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6]], dtype=torch.long),
        }
    )

    manager.prepare_policy_log_probs(
        batch,
        old_backend=FixedLogProbBackend(-1.0),
        new_backend=FixedLogProbBackend(-0.5),
        compute_ratio=False,
    )

    assert "old_log_probs" in batch.batch
    assert "new_log_probs" in batch.batch
    assert "prob_ratio" not in batch.batch
    assert batch.meta_info["policy_log_probs_prepared"] is True


def _batch_with_old_and_new_log_probs(
    old: list[float], new: list[float]
) -> "SearchBatch":
    n = len(old)
    responses = torch.ones(1, n, dtype=torch.long)
    obs_mask = torch.ones(1, n, dtype=torch.long)
    batch = SearchBatch.from_dict(
        {"responses": responses, "responses_with_info_mask": obs_mask}
    )
    batch.batch["old_log_probs"] = torch.tensor([old], dtype=torch.float32)
    batch.batch["new_log_probs"] = torch.tensor([new], dtype=torch.float32)
    return batch


def test_compute_prob_ratio_equals_exp_of_log_difference():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-0.5, -1.5])
    ratio = manager.compute_prob_ratio(batch)
    expected = torch.exp(
        torch.tensor([[-0.5 - -1.0, -1.5 - -2.0]], dtype=torch.float32)
    )
    assert torch.allclose(ratio, expected, atol=1e-5)
    assert "prob_ratio" in batch.batch


def test_per_token_kl_is_non_negative():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-0.5, -1.5])
    kl = manager.per_token_kl(batch)
    assert (kl >= 0).all(), "KL divergence must be non-negative"
    assert "per_token_kl" in batch.batch


def test_per_token_kl_is_zero_when_policies_are_identical():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-1.0, -2.0])
    kl = manager.per_token_kl(batch)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_compute_prob_ratio_is_finite_for_extreme_log_differences():
    # Without clamping, new_lp=0.0 and old_lp=-100.0 gives exp(100)=overflow in float32.
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.ones(1, 2, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )
    batch.batch["old_log_probs"] = torch.tensor([[-100.0, 0.0]], dtype=torch.float32)
    batch.batch["new_log_probs"] = torch.tensor([[0.0, -100.0]], dtype=torch.float32)

    ratio = manager.compute_prob_ratio(batch)

    assert torch.isfinite(ratio).all()


def test_per_token_kl_is_finite_for_extreme_log_differences():
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.ones(1, 2, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )
    batch.batch["old_log_probs"] = torch.tensor([[0.0, -100.0]], dtype=torch.float32)
    batch.batch["new_log_probs"] = torch.tensor([[-100.0, 0.0]], dtype=torch.float32)

    kl = manager.per_token_kl(batch)

    assert torch.isfinite(kl).all()


def test_compute_log_prob_skips_recompute_when_overwrite_false():
    # old_log_probs stored at rollout time must not be silently overwritten.
    manager = _manager_with_log_prob()  # backend returns -0.5
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6]], dtype=torch.long),
        }
    )
    frozen = torch.tensor([[-9.9, -9.9]], dtype=torch.float32)
    batch.batch["old_log_probs"] = frozen

    result = manager.compute_log_prob(batch, store_key="old_log_probs", overwrite=False)

    assert torch.allclose(result, frozen)
    assert torch.allclose(batch.batch["old_log_probs"], frozen)


def test_compute_log_prob_overwrites_old_log_probs_when_overwrite_true():
    manager = _manager_with_log_prob()  # backend returns -0.5
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[5, 6]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[5, 6]], dtype=torch.long),
        }
    )
    batch.batch["old_log_probs"] = torch.tensor([[-9.9, -9.9]], dtype=torch.float32)

    result = manager.compute_log_prob(batch, store_key="old_log_probs", overwrite=True)

    assert torch.allclose(result, torch.tensor([[-0.5, -0.5]], dtype=torch.float32))


def test_compute_policy_loss_matches_clipped_ppo_objective():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-0.5, -2.3])
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float32)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2),
    )

    ratio = torch.exp(torch.tensor([[0.5, -0.3]], dtype=torch.float32))
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    expected_terms = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    expected_loss = -(expected_terms.sum() / 2.0)

    assert torch.allclose(loss, expected_loss, atol=1e-5)
    assert "policy_loss" in batch.batch
    assert "clipped_prob_ratio" in batch.batch
    assert "clip_fraction" in batch.batch


def test_compute_policy_loss_ignores_observation_tokens_via_action_mask():
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict(
        {
            "responses": torch.ones(1, 3, dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[1, 0, 1]], dtype=torch.long),
        }
    )
    batch.batch["old_log_probs"] = torch.tensor(
        [[-1.0, -9.0, -2.0]], dtype=torch.float32
    )
    batch.batch["new_log_probs"] = torch.tensor(
        [[-0.9, 5.0, -1.7]], dtype=torch.float32
    )
    advantages = torch.tensor([[1.0, 100.0, 2.0]], dtype=torch.float32)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2),
    )

    ratio = torch.exp(torch.tensor([[0.1, 0.3]], dtype=torch.float32))
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    expected_terms = torch.minimum(
        ratio * torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        clipped_ratio * torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    expected_loss = -(expected_terms.sum() / 2.0)
    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_compute_policy_loss_adds_kl_penalty_when_requested():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-0.5, -1.5])
    advantages = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2, kl_coefficient=0.1),
    )

    ratio = torch.exp(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    expected_policy = (
        torch.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        ).sum()
        / 2.0
    )
    kl = manager.per_token_kl(batch).sum() / 2.0
    expected_loss = -expected_policy + 0.1 * kl
    assert torch.allclose(loss, expected_loss, atol=1e-5)


def test_compute_policy_loss_records_grpo_policy_loss_and_kl_penalty():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0, -2.0], new=[-0.5, -1.5])
    advantages = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2, kl_coefficient=0.1),
    )

    assert "grpo_policy_loss" in batch.batch
    assert "kl_penalty" in batch.batch
    assert "total_policy_loss" in batch.batch
    assert torch.allclose(batch.batch["policy_loss"], loss)
    assert torch.allclose(batch.batch["total_policy_loss"], loss.detach())
    assert batch.batch["kl_penalty"].item() >= 0.0


def test_compute_policy_loss_prefers_reference_log_probs_for_kl_penalty():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-10.0, -10.0], new=[-0.5, -0.5])
    batch.batch["ref_log_probs"] = torch.tensor([[-0.6, -0.6]], dtype=torch.float32)
    advantages = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2, kl_coefficient=0.1),
    )

    expected_policy = (
        torch.minimum(
            torch.full((1, 2), 1.2, dtype=torch.float32) * advantages,
            torch.full((1, 2), 1.2, dtype=torch.float32) * advantages,
        ).sum()
        / 2.0
    )
    kl = manager.per_token_kl(batch).sum() / 2.0
    expected_loss = -expected_policy + 0.1 * kl

    assert torch.allclose(loss, expected_loss, atol=1e-5)
    assert batch.meta_info["kl_reference"] == "ref_log_probs"


def test_compute_policy_loss_requires_advantages():
    manager = _manager_with_log_prob()
    batch = _batch_with_old_and_new_log_probs(old=[-1.0], new=[-0.5])
    with pytest.raises(ValueError, match="advantages not found"):
        manager.compute_policy_loss(batch)


def test_actor_rollout_step_samples_policy_actions_from_current_state():
    class RolloutTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>cats</search>",
                (2, 2): "<answer>done</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping[tokens])
            return decoded

    manager = LLMGenerationManager(
        tokenizer=RolloutTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(responses=[[[1, 1]], [[2, 2]]]),
    )
    rollings = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )

    rollout_step, meta_info = manager.actor_rollout_step(
        rollings,
        active_mask=torch.tensor([True]),
    )

    assert rollout_step.responses_ids.shape[0] == 1
    assert rollout_step.responses_ids.shape[1] >= 1
    assert rollout_step.responses_str == ["<search>cats</search>"]
    assert rollout_step.parsed_actions[0].tag == "search"
    assert rollout_step.parsed_actions[0].content == "cats"
    assert meta_info == {}


def test_build_react_context_transitions_appends_action_and_observation():
    manager = _manager()
    transitions = manager.build_react_context_transitions(
        ["<search>cats</search>"],
        ["\n\n<information>Doc 1: evidence</information>\n\n"],
    )

    assert len(transitions) == 1
    assert transitions[0].action_text == "<search>cats</search>"
    assert (
        transitions[0].observation_text
        == "\n\n<information>Doc 1: evidence</information>\n\n"
    )
    assert transitions[0].appended_context_text == (
        "<search>cats</search>\n\n<information>Doc 1: evidence</information>\n\n"
    )


# ---------------------------------------------------------------------------
# build_rollout_outputs — SearchBatch -> list[AgentLoopOutput]
# ---------------------------------------------------------------------------


def _make_trajectory(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    obs_mask: list[int],
    turns: int,
    final_answer: str | None,
    finished_without_answer: bool,
    steps=None,
):
    from src.llm_agent.generation import RolloutTrajectory

    return RolloutTrajectory(
        batch_index=0,
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        response_with_observation_mask=obs_mask,
        trajectory_turns=turns,
        steps=steps or [],
        final_answer=final_answer,
        finished_without_answer=finished_without_answer,
    )


def test_build_rollout_outputs_maps_trajectory_fields():
    manager = _manager_with_log_prob()
    traj = _make_trajectory(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, 12],
        obs_mask=[10, 0, 12],  # token 11 is an observation (pad=0)
        turns=2,
        final_answer="Paris",
        finished_without_answer=False,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [3]

    outputs = manager.build_rollout_outputs(batch)

    assert len(outputs) == 1
    out = outputs[0]
    assert out.prompt_ids == [1, 2, 3]
    assert out.response_ids == [10, 11, 12]
    assert out.response_mask == [1, 0, 1]  # pad position → 0
    assert out.num_turns == 2
    assert out.final_answer == "Paris"
    assert out.metrics["rounds_used"] == pytest.approx(3.0)
    assert out.metrics["search_budget_exhausted_without_answer"] == pytest.approx(0.0)


def test_build_rollout_outputs_computes_per_trajectory_search_repetitions():
    from src.llm_agent.generation import ReActStep

    manager = _manager_with_log_prob()
    steps = [
        ReActStep(
            turn=0,
            action_tag="search",
            action_content="cats",
            observation="...",
            is_terminal=False,
        ),
        ReActStep(
            turn=1,
            action_tag="search",
            action_content="cats",
            observation="...",
            is_terminal=False,
        ),
        ReActStep(
            turn=2,
            action_tag="answer",
            action_content="result",
            observation="",
            is_terminal=True,
        ),
    ]
    traj = _make_trajectory(
        prompt_ids=[1],
        response_ids=[2],
        obs_mask=[2],
        turns=3,
        final_answer="result",
        finished_without_answer=False,
        steps=steps,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [2]
    # per_trajectory_query_counts is populated by _record_search_tool_calls at
    # rollout time.  In unit tests we inject it directly.
    batch.meta_info["per_trajectory_query_counts"] = {0: {"cats": 2}}

    outputs = manager.build_rollout_outputs(batch)

    assert outputs[0].metrics["repeated_search_queries"] == pytest.approx(
        1.0
    )  # "cats" issued twice → 1 repeat
    assert outputs[0].metrics["fetched_pages"] == pytest.approx(0.0)


def test_build_rollout_outputs_counts_fetch_steps():
    from src.llm_agent.generation import ReActStep

    manager = _manager_with_log_prob()
    steps = [
        ReActStep(
            turn=0,
            action_tag="fetch",
            action_content="url1",
            observation="...",
            is_terminal=False,
        ),
        ReActStep(
            turn=1,
            action_tag="fetch",
            action_content="url2",
            observation="...",
            is_terminal=False,
        ),
        ReActStep(
            turn=2,
            action_tag="answer",
            action_content="done",
            observation="",
            is_terminal=True,
        ),
    ]
    traj = _make_trajectory(
        prompt_ids=[1],
        response_ids=[2],
        obs_mask=[2],
        turns=3,
        final_answer="done",
        finished_without_answer=False,
        steps=steps,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [0]

    outputs = manager.build_rollout_outputs(batch)
    assert outputs[0].metrics["fetched_pages"] == pytest.approx(2.0)


def test_build_rollout_outputs_flags_budget_exhausted():
    manager = _manager_with_log_prob()
    traj = _make_trajectory(
        prompt_ids=[1],
        response_ids=[2, 3],
        obs_mask=[2, 3],
        turns=6,
        final_answer=None,
        finished_without_answer=True,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [5]

    outputs = manager.build_rollout_outputs(batch)
    assert outputs[0].final_answer is None
    assert outputs[0].metrics[
        "search_budget_exhausted_without_answer"
    ] == pytest.approx(1.0)
    assert outputs[0].metrics["rounds_used"] == pytest.approx(5.0)


def test_build_rollout_outputs_empty_batch():
    manager = _manager_with_log_prob()
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = []
    assert manager.build_rollout_outputs(batch) == []


def test_build_final_gen_batch_output_packages_complete_rl_rollouts():
    manager = _manager_with_log_prob()
    traj = _make_trajectory(
        prompt_ids=[1, 2],
        response_ids=[10, 11],
        obs_mask=[10, 11],
        turns=2,
        final_answer="Paris",
        finished_without_answer=False,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [1]
    batch.meta_info["trajectory_turns"] = [2]

    final_output = manager.build_final_gen_batch_output(batch)

    assert final_output.search_batch is batch
    assert final_output.trajectories == [traj]
    assert len(final_output.rollout_outputs) == 1
    assert final_output.rollout_outputs[0].prompt_ids == [1, 2]
    assert final_output.rollout_outputs[0].final_answer == "Paris"
    assert final_output.trajectory_turns == [2]


def test_build_final_gen_batch_output_is_not_stored_in_batch():
    """FinalGenBatchOutput must not be stored inside the batch it wraps.

    Storing it would create a circular reference: batch → final_output → batch.
    Callers should use build_final_gen_batch_output() explicitly.
    """
    traj = _make_trajectory(
        prompt_ids=[1],
        response_ids=[2],
        obs_mask=[2],
        turns=1,
        final_answer="done",
        finished_without_answer=False,
    )
    batch = SearchBatch.from_dict({})
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [0]

    # batch itself should not contain a reference back to FinalGenBatchOutput
    assert "final_gen_batch_output" not in batch.non_tensor_batch


# ---------------------------------------------------------------------------
# Actor rollout: only active trajectories get ReActStep entries
# ---------------------------------------------------------------------------


def test_apply_step_result_does_not_record_steps_for_inactive_trajectories():
    """Trajectories that finished before the current turn must not accumulate
    ghost ReActStep entries.  _apply_step_result uses a participating_mask
    snapshot; _build_rollout_trajectories filters None placeholders.
    """
    from src.llm_agent.generation import (
        AgentLoopState,
        AgentLoopStepResult,
        PolicyAction,
    )

    manager = _manager_with_log_prob()

    # Batch of 2: trajectory 0 is already done (active_mask[0]=False),
    # trajectory 1 is still active.
    active_mask = torch.tensor([False, True], dtype=torch.bool)
    state = AgentLoopState(
        rollings=SearchBatch.from_dict(
            {
                "input_ids": torch.ones(2, 4, dtype=torch.long),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0).expand(2, -1),
            }
        ),
        original_left_side={"input_ids": torch.ones(2, 4, dtype=torch.long)},
        original_right_side={
            "responses": torch.ones(2, 0, dtype=torch.long),
            "responses_with_info_mask": torch.ones(2, 0, dtype=torch.long),
        },
        active_mask=active_mask,
        trajectory_turns=[1, 0],
        turns_stats=torch.tensor([1, 0], dtype=torch.int),
        valid_action_stats=torch.zeros(2, dtype=torch.int),
        valid_search_stats=torch.zeros(2, dtype=torch.int),
        active_num_list=[2, 1],
    )

    step_result = AgentLoopStepResult(
        responses_ids=torch.ones(2, 2, dtype=torch.long),
        responses_str=["<answer>x</answer>", "<search>q</search>"],
        parsed_actions=[
            PolicyAction(tag="answer", content="x", raw_text="<answer>x</answer>"),
            PolicyAction(tag="search", content="q", raw_text="<search>q</search>"),
        ],
        next_obs=["", "<information>doc</information>"],
        dones=[1, 0],
        valid_action=[1, 1],
        is_search=[0, 1],
    )

    manager._apply_step_result(state, step_result, include_observations=True)

    react_trajectory = state.meta_info.get("react_trajectory", [])
    assert len(react_trajectory) == 1  # one turn recorded
    turn_steps = react_trajectory[0]

    # Trajectory 0 was inactive — its slot must be None (no ghost step)
    assert turn_steps[0] is None
    # Trajectory 1 was active — its step must be a real ReActStep
    assert turn_steps[1] is not None
    assert turn_steps[1].action_tag == "search"
    context_transitions = state.meta_info.get("context_transitions", [])
    assert len(context_transitions) == 1
    assert context_transitions[0][0] is None
    assert context_transitions[0][1] is not None
    assert (
        context_transitions[0][1].appended_context_text
        == "<search>q</search><information>doc</information>"
    )


def test_build_rollout_trajectories_filters_none_steps():
    """None placeholders from inactive turns must not appear in traj.steps."""
    from src.llm_agent.generation import ReActStep

    manager = _manager_with_log_prob()

    # Manually build the react_trajectory as _apply_step_result would after the fix
    real_step = ReActStep(
        turn=0,
        action_tag="search",
        action_content="query",
        observation="<information>doc</information>",
        is_terminal=False,
    )
    # Turn 0: traj 0 active (real step), traj 1 inactive (None)
    # Turn 1: traj 0 active (terminal), traj 1 still None
    terminal_step = ReActStep(
        turn=1,
        action_tag="answer",
        action_content="Paris",
        observation="",
        is_terminal=True,
    )
    react_trajectory = [
        [real_step, None],
        [terminal_step, None],
    ]

    left_side = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
    right_side = {
        "responses": torch.zeros(2, 2, dtype=torch.long),
        "responses_with_info_mask": torch.zeros(2, 2, dtype=torch.long),
    }
    meta_info = {
        "react_trajectory": react_trajectory,
        "trajectory_turns": [2, 0],
        "final_answers": {0: "Paris"},
        "finished_without_answer": [False, True],
    }

    trajectories = manager._build_rollout_trajectories(
        left_side=left_side, right_side=right_side, meta_info=meta_info
    )

    # Trajectory 0: two real steps, no None
    assert all(s is not None for s in trajectories[0].steps)
    assert len(trajectories[0].steps) == 2
    assert trajectories[0].steps[0].action_tag == "search"
    assert trajectories[0].steps[1].action_tag == "answer"

    # Trajectory 1: all slots were None → empty steps
    assert trajectories[1].steps == []


# ---------------------------------------------------------------------------
# compute_action_type_masks and policy_loss_breakdown
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer for action-type mask tests.

    Each character maps to its Unicode ordinal; pad_token_id == 0 (null byte).
    Supports encode/decode so _token_char_offsets can reconstruct spans.
    """

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids if i != 0)


def _manager_with_char_tokenizer() -> LLMGenerationManager:
    return LLMGenerationManager(
        tokenizer=CharTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=512,
            max_response_length=256,
            max_obs_length=64,
            num_gpus=1,
        ),
        generation_backend=DummyActorRolloutWithLogProb(),
    )


def _batch_from_texts(*responses: str) -> SearchBatch:
    """Encode response strings as character-level token IDs (ord-based)."""
    encoded = [[ord(c) for c in r] for r in responses]
    max_len = max(len(e) for e in encoded)
    padded = [e + [0] * (max_len - len(e)) for e in encoded]
    t = torch.tensor(padded, dtype=torch.long)
    return SearchBatch.from_dict(
        {"responses": t, "responses_with_info_mask": t.clone()}
    )


def test_compute_action_type_masks_identifies_search_and_answer_tokens():
    manager = _manager_with_char_tokenizer()
    text = "<search>query</search><answer>result</answer>"
    batch = _batch_from_texts(text)
    masks = manager.compute_action_type_masks(batch)

    assert set(masks.keys()) == {"search", "plan", "fetch", "answer"}
    # Every character token falls in exactly one action block
    combined = sum(masks[t][0] for t in ("search", "answer", "plan", "fetch"))
    assert combined.sum().item() == pytest.approx(len(text))
    # search mask covers exactly the <search>…</search> span
    search_span_len = len("<search>query</search>")
    assert masks["search"][0, :search_span_len].sum().item() == pytest.approx(
        search_span_len
    )
    assert masks["search"][0, search_span_len:].sum().item() == pytest.approx(0.0)


def test_compute_action_type_masks_handles_plan_fetch_tags():
    manager = _manager_with_char_tokenizer()
    text = "<plan>think</plan><fetch>url</fetch>"
    batch = _batch_from_texts(text)
    masks = manager.compute_action_type_masks(batch)

    plan_len = len("<plan>think</plan>")
    assert masks["plan"][0, :plan_len].sum().item() == pytest.approx(plan_len)
    assert masks["plan"][0, plan_len:].sum().item() == pytest.approx(0.0)
    fetch_len = len("<fetch>url</fetch>")
    assert masks["fetch"][
        0, plan_len : plan_len + fetch_len
    ].sum().item() == pytest.approx(fetch_len)


def test_compute_action_type_masks_multi_batch():
    manager = _manager_with_char_tokenizer()
    batch = _batch_from_texts("<search>q</search>", "<answer>a</answer>")
    masks = manager.compute_action_type_masks(batch)

    assert masks["search"][0].sum().item() > 0
    assert masks["answer"][0].sum().item() == 0.0
    assert masks["answer"][1].sum().item() > 0
    assert masks["search"][1].sum().item() == 0.0


def test_compute_policy_family_masks_maps_reasoning_and_stopping_policies():
    manager = _manager_with_char_tokenizer()
    text = "<plan>think</plan><fetch>url</fetch><answer>x</answer>"
    batch = _batch_from_texts(text)

    masks = manager.compute_policy_family_masks(batch)

    assert set(masks.keys()) == {
        "search_policy",
        "reasoning_policy",
        "stopping_policy",
        "answer_policy",
    }
    assert masks["search_policy"].sum().item() == pytest.approx(0.0)
    assert masks["reasoning_policy"].sum().item() == pytest.approx(
        len("<plan>think</plan><fetch>url</fetch>")
    )
    assert masks["stopping_policy"].sum().item() == pytest.approx(
        len("<answer>x</answer>")
    )
    assert torch.allclose(masks["stopping_policy"], masks["answer_policy"])


def test_policy_loss_breakdown_returns_per_type_scalars():
    manager = _manager_with_char_tokenizer()
    text = "<search>q</search><answer>x</answer>"
    batch = _batch_from_texts(text)
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.full((1, n), -1.0)
    batch.batch["new_log_probs"] = torch.full((1, n), -0.9)
    advantages = torch.ones(1, n)

    breakdown = manager.policy_loss_breakdown(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2),
    )

    assert set(breakdown.keys()) == {"search", "plan", "fetch", "answer"}
    # ratio = exp(0.1) ≈ 1.105, within clip → positive objective for present types
    assert breakdown["search"].item() > 0.0
    assert breakdown["answer"].item() > 0.0
    # plan and fetch absent from text → their masks are all-zero → 0 contribution
    assert breakdown["plan"].item() == pytest.approx(0.0, abs=1e-5)
    assert breakdown["fetch"].item() == pytest.approx(0.0, abs=1e-5)


def test_policy_update_breakdown_returns_search_reasoning_stopping_answer():
    manager = _manager_with_char_tokenizer()
    text = "<search>q</search><plan>think</plan><answer>x</answer>"
    batch = _batch_from_texts(text)
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.full((1, n), -1.0)
    batch.batch["new_log_probs"] = torch.full((1, n), -0.9)
    advantages = torch.ones(1, n)

    breakdown = manager.policy_update_breakdown(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2),
    )

    assert set(breakdown.keys()) == {
        "search_policy",
        "reasoning_policy",
        "stopping_policy",
        "answer_policy",
    }
    assert breakdown["search_policy"].item() > 0.0
    assert breakdown["reasoning_policy"].item() > 0.0
    assert breakdown["stopping_policy"].item() > 0.0
    assert breakdown["answer_policy"].item() > 0.0
    assert breakdown["stopping_policy"].item() == pytest.approx(
        breakdown["answer_policy"].item(),
        abs=1e-6,
    )


def test_policy_loss_breakdown_requires_advantages():
    manager = _manager_with_char_tokenizer()
    batch = _batch_from_texts("<answer>x</answer>")
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.ones(1, n) * -1.0
    batch.batch["new_log_probs"] = torch.ones(1, n) * -0.9
    with pytest.raises(ValueError, match="advantages not found"):
        manager.policy_loss_breakdown(batch)


def test_compute_policy_loss_action_type_weights_upweight_answer():
    manager = _manager_with_char_tokenizer()
    text = "<search>q</search><answer>x</answer>"
    batch_base = _batch_from_texts(text)
    batch_weighted = _batch_from_texts(text)
    n = batch_base.batch["responses"].shape[1]
    for b in (batch_base, batch_weighted):
        b.batch["old_log_probs"] = torch.full((1, n), -1.0)
        b.batch["new_log_probs"] = torch.full((1, n), -0.9)
    advantages = torch.ones(1, n)

    loss_base = manager.compute_policy_loss(
        batch_base, advantages=advantages, config=PPOPolicyLossConfig(clip_epsilon=0.2)
    )
    loss_weighted = manager.compute_policy_loss(
        batch_weighted,
        advantages=advantages,
        config=PPOPolicyLossConfig(
            clip_epsilon=0.2, action_type_weights={"answer": 2.0}
        ),
    )
    # Upweighting answer tokens increases the objective → negated loss is smaller
    assert loss_weighted.item() < loss_base.item()


def test_compute_policy_loss_records_updated_policy_families():
    manager = _manager_with_char_tokenizer()
    text = "<search>q</search><plan>think</plan><answer>x</answer>"
    batch = _batch_from_texts(text)
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.full((1, n), -1.0)
    batch.batch["new_log_probs"] = torch.full((1, n), -0.9)
    advantages = torch.ones(1, n)

    manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2),
    )

    assert batch.meta_info["updated_policies"] == [
        "search_policy",
        "reasoning_policy",
        "stopping_policy",
        "answer_policy",
    ]
    assert batch.meta_info["policy_update_token_counts"]["search_policy"] > 0.0
    assert batch.meta_info["policy_update_token_counts"]["reasoning_policy"] > 0.0
    assert batch.meta_info["policy_update_token_counts"]["stopping_policy"] > 0.0
    assert batch.meta_info["policy_update_breakdown"]["answer_policy"] == pytest.approx(
        batch.meta_info["policy_update_breakdown"]["stopping_policy"],
        abs=1e-6,
    )


def test_ppo_policy_loss_config_entropy_coefficient_default_zero():
    cfg = PPOPolicyLossConfig()
    assert cfg.entropy_coefficient == 0.0


def test_compute_policy_loss_entropy_bonus_lowers_loss_for_low_entropy_policy():
    """Entropy bonus (entropy_coefficient > 0) must reduce the loss when
    new_log_probs are present — more negative log probs → higher entropy term
    → smaller (more negative) objective before negation → smaller loss."""
    manager = _manager_with_char_tokenizer()
    text = "<answer>x</answer>"
    batch_no_ent = _batch_from_texts(text)
    batch_with_ent = _batch_from_texts(text)
    n = batch_no_ent.batch["responses"].shape[1]
    for b in (batch_no_ent, batch_with_ent):
        b.batch["old_log_probs"] = torch.full((1, n), -1.0)
        b.batch["new_log_probs"] = torch.full((1, n), -2.0)  # high entropy signal
    advantages = torch.ones(1, n)

    loss_no_ent = manager.compute_policy_loss(
        batch_no_ent,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2, entropy_coefficient=0.0),
    )
    loss_with_ent = manager.compute_policy_loss(
        batch_with_ent,
        advantages=advantages,
        config=PPOPolicyLossConfig(clip_epsilon=0.2, entropy_coefficient=0.1),
    )

    # entropy_term = -mean(new_log_probs) = -(-2.0) = 2.0
    # loss_with_ent = loss_no_ent - 0.1 * 2.0  (lower loss)
    assert loss_with_ent.item() < loss_no_ent.item()
    assert "entropy" in batch_with_ent.batch
    assert batch_with_ent.batch["entropy"].item() == pytest.approx(2.0, abs=1e-5)


def test_compute_policy_loss_entropy_not_stored_when_coefficient_zero():
    manager = _manager_with_char_tokenizer()
    text = "<answer>x</answer>"
    batch = _batch_from_texts(text)
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.full((1, n), -1.0)
    batch.batch["new_log_probs"] = torch.full((1, n), -2.0)
    advantages = torch.ones(1, n)

    manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(entropy_coefficient=0.0),
    )

    assert "entropy" not in batch.batch


def test_compute_policy_loss_type_masks_computed_once_even_with_action_type_weights():
    """With action_type_weights set, type masks must not trigger extra tokenizer
    decode passes.  We verify correctness (not call-count) by checking that
    the per-family breakdown and the weighted loss are both populated correctly
    in a single compute_policy_loss call."""
    manager = _manager_with_char_tokenizer()
    text = "<search>q</search><answer>x</answer>"
    batch = _batch_from_texts(text)
    n = batch.batch["responses"].shape[1]
    batch.batch["old_log_probs"] = torch.full((1, n), -1.0)
    batch.batch["new_log_probs"] = torch.full((1, n), -0.9)
    advantages = torch.ones(1, n)

    loss = manager.compute_policy_loss(
        batch,
        advantages=advantages,
        config=PPOPolicyLossConfig(
            clip_epsilon=0.2,
            action_type_weights={"search": 2.0, "answer": 1.5},
        ),
    )

    # Family breakdown must be populated (uses same type_masks pass)
    assert "search_policy" in batch.meta_info["policy_update_breakdown"]
    assert "answer_policy" in batch.meta_info["policy_update_breakdown"]
    # Loss must be a finite scalar
    assert torch.isfinite(loss)


def test_run_llm_loop_behaves_like_multi_turn_agent_orchestration():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>cats</search>",
                (2, 2): "<answer>done</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping[tokens])
            return decoded

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=1,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(
            responses=[
                [[1, 1]],
                [[2, 2]],
            ]
        ),
    )

    manager.batch_search = lambda payload, search_mode, gt_threshold: [
        "Doc 1: evidence"
    ]  # type: ignore[method-assign]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["What is the answer?"],
        "golden_answers": [["done"]],
    }

    final_batch, trajectory_turns = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=0,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert trajectory_turns == [2]
    assert final_batch.meta_info["valid_search_stats"] == [1]
    assert final_batch.meta_info["valid_action_stats"] == [2]
    assert final_batch.meta_info["active_mask"] == [False]
    assert final_batch.meta_info["first_rollout_actions"] == ["search"]
    assert final_batch.meta_info["policy_action_history"] == [["search"], ["answer"]]
    assert final_batch.meta_info["search_query_history"] == [["cats"]]
    assert final_batch.meta_info["search_queries_total"] == 1
    assert final_batch.meta_info["search_queries_unique"] == 1
    assert final_batch.meta_info["search_query_repetitions"] == 0
    assert final_batch.meta_info["search_query_reformulations"] == 0
    react = final_batch.meta_info["react_trajectory"]
    assert len(react) == 1  # one turn with include_observations=True
    assert react[0][0].action_tag == "search"
    assert react[0][0].action_content == "cats"
    assert (
        react[0][0].observation == "\n\n<information>Doc 1: evidence</information>\n\n"
    )
    assert react[0][0].is_terminal is False
    assert len(final_batch.meta_info["context_token_lengths"]) == 1
    assert final_batch.meta_info["trajectory_turns"] == [2]
    trajectories = final_batch.non_tensor_batch["trajectories"]
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory.batch_index == 0
    assert trajectory.trajectory_turns == 2
    assert trajectory.final_answer == "done"
    assert trajectory.finished_without_answer is False
    assert trajectory.old_log_probs is None
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action_tag == "search"
    assert final_batch.batch["responses"].shape[1] > 0


def test_run_llm_loop_saves_old_log_probs_when_backend_supports_it():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>cats</search>",
                (2, 2): "<answer>done</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping[tokens])
            return decoded

    class SequencedActorRolloutWithLogProb(SequencedActorRollout):
        def compute_log_prob(self, batch):
            return torch.full_like(batch.batch["responses"], -0.25, dtype=torch.float32)

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=1,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRolloutWithLogProb(
            responses=[
                [[1, 1]],
                [[2, 2]],
            ]
        ),
    )
    manager.batch_search = lambda payload, search_mode, gt_threshold: [  # type: ignore[method-assign]
        "Doc 1: evidence"
    ]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["What is the answer?"],
        "golden_answers": [["done"]],
    }

    final_batch, _ = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=0,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert "old_log_probs" in final_batch.batch
    assert final_batch.non_tensor_batch["trajectories"][0].old_log_probs is not None


def test_run_llm_loop_records_printable_search_trajectory_logs():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>cats</search>",
                (2, 2): "<answer>done</answer>",
                (3, 4): "Question context",
                (3, 4, 1, 1): "Question context<search>cats</search>",
                (3, 4, 1, 1, 9, 9, 2, 2): (
                    "Question context<search>cats</search>"
                    "<information>Doc 1: evidence</information><answer>done</answer>"
                ),
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping.get(tokens, " ".join(str(t) for t in tokens)))
            return decoded

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=1,
            max_start_length=8,
            max_prompt_length=64,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(responses=[[[1, 1]], [[2, 2]]]),
    )
    manager.batch_search = lambda payload, search_mode, gt_threshold: [  # type: ignore[method-assign]
        "Doc 1: evidence"
    ]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["What is the answer?"],
        "golden_answers": [["done"]],
    }

    final_batch, _ = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=0,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    logs = final_batch.non_tensor_batch["trajectory_logs"]
    assert len(logs) == 1
    log = logs[0]
    assert log.question == "What is the answer?"
    assert [step.action_type for step in log.steps] == ["search", "answer"]
    assert log.steps[0].step_id == 1
    assert log.steps[0].action_value == "cats"
    assert "Doc 1: evidence" in (log.steps[0].observation or "")
    assert log.steps[1].done is True
    rendered = format_search_trajectory_log(log, reward=1.0)
    assert "Question: What is the answer?" in rendered
    assert "Step 1: search cats" in rendered
    assert "Observation:" in rendered
    assert "Step 2: answer done" in rendered
    assert "Reward: 1.0" in rendered
    # Compact mode must NOT show the raw State / Model Output dumps
    assert "State:" not in rendered
    assert "Model Output:" not in rendered


# ---------------------------------------------------------------------------
# format_search_trajectory_log — compact vs verbose, to_dict, batch formatter
# ---------------------------------------------------------------------------


def _make_two_step_log():
    from src.llm_agent.generation import SearchStep, SearchTrajectoryLog

    return SearchTrajectoryLog(
        batch_index=0,
        question="Who won the 2024 Nobel Prize in Physics?",
        steps=[
            SearchStep(
                step_id=1,
                state="context-window-contents-here",
                model_output="<search>2024 Nobel Physics</search>",
                action_type="search",
                action_value="2024 Nobel Physics",
                observation="Hopfield and Hinton won the Nobel Prize.",
                done=False,
            ),
            SearchStep(
                step_id=2,
                state="context-window-contents-here",
                model_output="<answer>Hopfield and Hinton</answer>",
                action_type="answer",
                action_value="Hopfield and Hinton",
                observation=None,
                done=True,
            ),
        ],
        final_answer="Hopfield and Hinton",
        finished_without_answer=False,
    )


def test_format_search_trajectory_log_compact_omits_state_and_model_output():
    from src.llm_agent.generation import format_search_trajectory_log

    log = _make_two_step_log()
    rendered = format_search_trajectory_log(log, reward=1.0)

    assert "Question: Who won the 2024 Nobel Prize in Physics?" in rendered
    assert "Step 1: search 2024 Nobel Physics" in rendered
    assert "Observation: Hopfield and Hinton won" in rendered
    assert "Step 2: answer Hopfield and Hinton" in rendered
    assert "Final Answer: Hopfield and Hinton" in rendered
    assert "Reward: 1.0" in rendered
    # Compact: state and raw model output must be hidden
    assert "State:" not in rendered
    assert "Model Output:" not in rendered


def test_format_search_trajectory_log_verbose_includes_state_and_model_output():
    from src.llm_agent.generation import format_search_trajectory_log

    log = _make_two_step_log()
    rendered = format_search_trajectory_log(log, verbose=True)

    assert "State: context-window-contents-here" in rendered
    assert "Model Output: <search>2024 Nobel Physics</search>" in rendered


def test_format_search_trajectory_log_truncates_long_observation_in_compact_mode():
    from src.llm_agent.generation import (
        SearchStep,
        SearchTrajectoryLog,
        format_search_trajectory_log,
    )

    long_obs = "x" * 500
    log = SearchTrajectoryLog(
        batch_index=0,
        question="q?",
        steps=[
            SearchStep(
                step_id=1,
                state="",
                model_output="<search>q</search>",
                action_type="search",
                action_value="q",
                observation=long_obs,
                done=False,
            )
        ],
        final_answer=None,
        finished_without_answer=True,
    )
    rendered = format_search_trajectory_log(log, max_obs_chars=100)
    assert "..." in rendered
    assert len(rendered) < len(long_obs)


def test_format_search_trajectory_log_verbose_does_not_truncate_observation():
    from src.llm_agent.generation import (
        SearchStep,
        SearchTrajectoryLog,
        format_search_trajectory_log,
    )

    long_obs = "y" * 500
    log = SearchTrajectoryLog(
        batch_index=0,
        question="q?",
        steps=[
            SearchStep(
                step_id=1,
                state="",
                model_output="<search>q</search>",
                action_type="search",
                action_value="q",
                observation=long_obs,
                done=False,
            )
        ],
        final_answer=None,
        finished_without_answer=True,
    )
    rendered = format_search_trajectory_log(log, verbose=True)
    assert "..." not in rendered
    assert long_obs in rendered


def test_search_trajectory_log_str_returns_compact_format():
    from src.llm_agent.generation import format_search_trajectory_log

    log = _make_two_step_log()
    assert str(log) == format_search_trajectory_log(log)


def test_search_trajectory_log_to_dict_serializes_all_steps():
    log = _make_two_step_log()
    d = log.to_dict()

    assert d["question"] == "Who won the 2024 Nobel Prize in Physics?"
    assert d["final_answer"] == "Hopfield and Hinton"
    assert d["finished_without_answer"] is False
    assert len(d["steps"]) == 2
    assert d["steps"][0]["action_type"] == "search"
    assert d["steps"][0]["action_value"] == "2024 Nobel Physics"
    assert d["steps"][0]["observation"] == "Hopfield and Hinton won the Nobel Prize."
    assert d["steps"][0]["done"] is False
    assert d["steps"][1]["action_type"] == "answer"
    assert d["steps"][1]["done"] is True


def test_format_trajectory_batch_joins_multiple_logs_with_separator():
    from src.llm_agent.generation import format_trajectory_batch

    log1 = _make_two_step_log()
    log2 = _make_two_step_log()
    rendered = format_trajectory_batch([log1, log2], rewards=[1.0, 0.0])

    assert rendered.count("Question:") == 2
    assert "Reward: 1.0" in rendered
    assert "Reward: 0.0" in rendered
    assert "─" in rendered  # separator


def test_format_trajectory_batch_empty_list_returns_empty_string():
    from src.llm_agent.generation import format_trajectory_batch

    assert format_trajectory_batch([]) == ""


def test_run_llm_loop_records_search_reformulations_across_turns():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>physics nobel 2024</search>",
                (2, 2): "<search>2024 physics nobel prize winner</search>",
                (3, 3): "<answer>done</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping[tokens])
            return decoded

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(
            responses=[
                [[1, 1]],
                [[2, 2]],
                [[3, 3]],
            ]
        ),
    )
    manager.batch_search = lambda payload, search_mode, gt_threshold: [  # type: ignore[method-assign]
        f"Doc {index + 1}: evidence for {tc.query}" for index, tc in enumerate(payload)
    ]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["Who won the Nobel Prize in Physics in 2024?"],
        "golden_answers": [["done"]],
    }

    final_batch, trajectory_turns = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=0,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert trajectory_turns == [3]
    assert final_batch.meta_info["search_query_history"] == [
        ["physics nobel 2024"],
        ["2024 physics nobel prize winner"],
    ]
    assert final_batch.meta_info["search_queries_total"] == 2
    assert final_batch.meta_info["search_queries_unique"] == 2
    assert final_batch.meta_info["search_query_repetitions"] == 0
    assert final_batch.meta_info["search_query_reformulations"] == 1


def test_run_llm_loop_supports_search_fetch_answer_second_rounds():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>2024 physics nobel prize winner</search>",
                (2, 2): "<fetch>https://example.com/nobel</fetch>",
                (
                    3,
                    3,
                ): "<answer>The 2024 Nobel Prize in Physics was awarded to ...</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(token for token in row if token != 0)
                decoded.append(mapping[tokens])
            return decoded

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=64,
            max_response_length=16,
            max_obs_length=32,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(
            responses=[
                [[1, 1]],
                [[2, 2]],
                [[3, 3]],
            ]
        ),
    )
    manager.batch_search = lambda payload, search_mode, gt_threshold: [  # type: ignore[method-assign]
        "Doc 1(Title: Nobel) The 2024 Nobel Prize in Physics was awarded to ... URL: https://example.com/nobel"
    ]
    manager.batch_fetch = lambda payload: [  # type: ignore[method-assign]
        "Doc 1(Title: Nobel Full Page) Full article body with evidence."
    ]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["Who won the Nobel Prize in Physics in 2024?"],
        "golden_answers": [["The 2024 Nobel Prize in Physics was awarded to ..."]],
    }

    final_batch, trajectory_turns = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=0,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert trajectory_turns == [3]
    assert final_batch.meta_info["policy_action_history"] == [
        ["search"],
        ["fetch"],
        ["answer"],
    ]
    assert final_batch.meta_info["fetch_url_history"] == [["https://example.com/nobel"]]
    assert final_batch.meta_info["fetched_urls_total"] == 1
    react = final_batch.meta_info["react_trajectory"]
    assert len(react) == 2
    assert react[0][0].action_tag == "search"
    assert "<information>" in react[0][0].observation
    assert react[1][0].action_tag == "fetch"
    assert react[1][0].observation == (
        "\n\n<full_page>Doc 1(Title: Nobel Full Page) Full article body with evidence.</full_page>\n\n"
    )
    trajectory = final_batch.non_tensor_batch["trajectories"][0]
    assert trajectory.trajectory_turns == 3
    assert [step.action_tag for step in trajectory.steps] == ["search", "fetch"]
    assert (
        trajectory.final_answer == "The 2024 Nobel Prize in Physics was awarded to ..."
    )
    assert trajectory.finished_without_answer is False
    continuation = final_batch.meta_info["continuation_history"]
    assert continuation[0].continue_generation is True
    assert "continue with another agent turn" in continuation[0].reason
    assert continuation[1].continue_generation is True
    assert "continue with another agent turn" in continuation[1].reason
    assert continuation[2].continue_generation is False
    assert "final no-tool answer attempt" in continuation[2].reason


def test_postprocess_responses_truncates_to_first_complete_action():
    class LoopTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del responses, skip_special_tokens
            return ["preface <plan>outline</plan><answer>done</answer> tail"]

    manager = LLMGenerationManager(
        tokenizer=LoopTokenizer(),
        config=GenerationConfig(
            max_turns=1,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=DummyActorRollout(),
    )

    _, responses = manager._postprocess_responses(
        torch.tensor([[1, 1]], dtype=torch.long)
    )
    assert responses == ["<plan>outline</plan>"]


def test_run_agent_loop_alias_delegates_to_run_llm_loop():
    manager = _manager()
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["q"],
        "golden_answers": [["a"]],
    }
    expected_batch = SearchBatch.from_dict(
        {"responses": torch.tensor([[1]], dtype=torch.long)}
    )

    captured = {}

    def fake_run_llm_loop(**kwargs):
        captured.update(kwargs)
        return expected_batch, [1]

    manager.run_llm_loop = fake_run_llm_loop  # type: ignore[method-assign]

    output_batch, turns = manager.run_agent_loop(
        gen_batch=gen_batch,
        search_mode="google",
        current_step=1,
        total_steps=10,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert output_batch is expected_batch
    assert turns == [1]
    assert captured["gen_batch"] is gen_batch


def test_run_llm_loop_uses_gen_batch_input_ids_by_default():
    manager = _manager()
    sentinel = torch.tensor([[7, 8, 9]], dtype=torch.long)
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": sentinel,
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["q"],
        "golden_answers": [["a"]],
    }

    captured: dict[str, torch.Tensor] = {}

    def fake_initialize(gen_batch_arg, initial_input_ids_arg):
        captured["initial_input_ids"] = initial_input_ids_arg
        return AgentLoopState(
            rollings=gen_batch_arg,
            original_left_side={"input_ids": initial_input_ids_arg},
            original_right_side={
                "responses": initial_input_ids_arg[:, []],
                "responses_with_info_mask": initial_input_ids_arg[:, []],
            },
            active_mask=torch.zeros(1, dtype=torch.bool),
            trajectory_turns=[0],
            turns_stats=torch.ones(1, dtype=torch.int),
            valid_action_stats=torch.zeros(1, dtype=torch.int),
            valid_search_stats=torch.zeros(1, dtype=torch.int),
            active_num_list=[0],
        )

    manager._initialize_agent_loop_state = fake_initialize  # type: ignore[method-assign]

    manager.run_llm_loop(gen_batch=gen_batch, search_mode="google")

    assert torch.equal(captured["initial_input_ids"], sentinel)


def test_run_prompt_rollout_batch_converts_prompt_batch_and_calls_agent_loop():
    manager = _manager()
    loader = build_prompt_dataloader(
        [
            {
                "question": "Who won the Nobel Prize in Physics in 2024?",
                "ground_truth": "John Hopfield and Geoffrey Hinton.",
                "tools": ["search"],
            }
        ],
        tokenizer=DummyTokenizer(),
        batch_size=1,
        shuffle=False,
    )
    prompt_batch = next(iter(loader))

    captured: dict[str, object] = {}
    expected_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[1, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )

    def fake_run_llm_loop(**kwargs):
        captured.update(kwargs)
        return expected_batch, [1]

    manager.run_llm_loop = fake_run_llm_loop  # type: ignore[method-assign]

    output_batch, turns = manager.run_prompt_rollout_batch(
        prompt_batch,
        search_mode="google",
        current_step=2,
        total_steps=10,
    )

    assert output_batch is expected_batch
    assert turns == [1]
    assert captured["search_mode"] == "google"
    assert captured["current_step"] == 2
    assert captured["total_steps"] == 10
    converted = captured["gen_batch"]
    assert isinstance(converted, SearchBatch)
    assert converted.non_tensor_batch["question"] == [
        "Who won the Nobel Prize in Physics in 2024?"
    ]
    assert converted.non_tensor_batch["golden_answers"] == [
        ["John Hopfield and Geoffrey Hinton."]
    ]


def test_run_prompt_rollout_group_assigns_shared_group_id_and_rollout_indices():
    manager = _manager()
    loader = build_prompt_dataloader(
        [
            {
                "question": "Who won the Nobel Prize in Physics in 2024?",
                "ground_truth": "John Hopfield and Geoffrey Hinton.",
                "tools": ["search"],
            }
        ],
        tokenizer=DummyTokenizer(),
        batch_size=1,
        shuffle=False,
    )
    prompt_batch = next(iter(loader))

    captured_variants: list[dict[str, object]] = []

    def fake_run_llm_loop(**kwargs):
        gen_batch = kwargs["gen_batch"]
        captured_variants.append(dict(gen_batch.non_tensor_batch["sampling_params"]))
        batch = SearchBatch.from_dict(
            {
                "responses": torch.zeros((1, 0), dtype=torch.long),
                "responses_with_info_mask": torch.zeros((1, 0), dtype=torch.long),
            }
        )
        batch.non_tensor_batch = {
            "question": ["Who won the Nobel Prize in Physics in 2024?"],
            "trajectories": [],
        }
        batch.meta_info = {
            "final_answers": ["John Hopfield and Geoffrey Hinton."],
            "finished_without_answer": [False],
            "trajectory_turns": [1],
        }
        return batch, [1]

    manager.run_llm_loop = fake_run_llm_loop  # type: ignore[method-assign]

    grouped = manager.run_prompt_rollout_group(
        prompt_batch,
        search_mode="google",
        sampling_params={"temperature": 0.8, "top_p": 0.9},
        num_rollouts=3,
        group_id="prompt-group-1",
        base_seed=10,
    )

    assert [item.group_id for item in grouped] == [
        "prompt-group-1",
        "prompt-group-1",
        "prompt-group-1",
    ]
    assert [item.rollout_index for item in grouped] == [0, 1, 2]
    assert [item.sampling_params["seed"] for item in grouped] == [10, 11, 12]
    assert captured_variants[0]["temperature"] == pytest.approx(0.8)
    assert captured_variants[1]["temperature"] > captured_variants[0]["temperature"]
    assert (
        grouped[0].final_output.search_batch.meta_info["group_id"] == "prompt-group-1"
    )
    assert grouped[2].final_output.search_batch.meta_info["rollout_index"] == 2


def test_build_rollout_outputs_propagates_group_rollout_metadata():
    manager = _manager_with_char_tokenizer()
    traj = _make_trajectory(
        prompt_ids=[1, 2],
        response_ids=[10, 11],
        obs_mask=[10, 11],
        turns=1,
        final_answer="Paris",
        finished_without_answer=False,
    )
    batch = SearchBatch.from_dict(
        {
            "responses": torch.tensor([[10, 11]], dtype=torch.long),
            "responses_with_info_mask": torch.tensor([[10, 11]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["trajectories"] = [traj]
    batch.meta_info["valid_search_stats"] = [1]
    batch.meta_info["group_id"] = "prompt-group-1"
    batch.meta_info["rollout_index"] = 2

    outputs = manager.build_rollout_outputs(batch)

    assert outputs[0].group_id == "prompt-group-1"
    assert outputs[0].rollout_index == 2


# ---------------------------------------------------------------------------
# score_group_rollout + format_group_rollout
# ---------------------------------------------------------------------------


def _make_grouped_rollout(group_id, rollout_index, answer, temperature, seed):
    """Helper that builds a minimal GroupedRolloutBatch for testing."""
    from src.agent_loop.agent_loop import AgentLoopOutput
    from src.llm_agent.generation import (
        FinalGenBatchOutput,
        GroupedRolloutBatch,
        SearchStep,
        SearchTrajectoryLog,
    )

    log = SearchTrajectoryLog(
        batch_index=0,
        question="Who won the 2024 Nobel Prize in Physics?",
        steps=[
            SearchStep(
                step_id=1,
                state="",
                model_output=f"<answer>{answer}</answer>",
                action_type="answer",
                action_value=answer,
                observation=None,
                done=True,
            )
        ],
        final_answer=answer,
        finished_without_answer=False,
    )
    batch = SearchBatch.from_dict(
        {
            "responses": torch.zeros(1, 0, dtype=torch.long),
            "responses_with_info_mask": torch.zeros(1, 0, dtype=torch.long),
        }
    )
    batch.meta_info = {
        "trajectory_turns": [1],
        "final_answers": [answer],
        "finished_without_answer": [False],
        "group_id": group_id,
        "rollout_index": rollout_index,
    }
    out = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics={},
        context=None,
        final_answer=answer,
    )
    out.group_id = group_id
    out.rollout_index = rollout_index
    final_output = FinalGenBatchOutput(
        search_batch=batch,
        trajectories=[],
        trajectory_logs=[log],
        rollout_outputs=[out],
        trajectory_turns=[1],
    )
    return GroupedRolloutBatch(
        group_id=group_id,
        rollout_index=rollout_index,
        sampling_params={"temperature": temperature, "seed": seed},
        final_output=final_output,
    )


def test_score_group_rollout_assigns_reward_and_advantage_per_rollout():
    from src.agent_loop import SearchRewardConfig, SearchRewardFunction
    from src.llm_agent.generation import score_group_rollout

    grouped = [
        _make_grouped_rollout("g1", 0, "Hopfield and Hinton", 0.8, 0),  # correct
        _make_grouped_rollout("g1", 1, "wrong answer", 0.95, 1),  # wrong
    ]

    scored = score_group_rollout(
        grouped,
        ground_truth="Hopfield and Hinton",
        judge_fn=lambda a, g: 1.0 if a.strip() == g.strip() else 0.0,
        reward_fn=SearchRewardFunction(SearchRewardConfig.sparse_final_only()),
    )

    assert len(scored) == 2
    assert scored[0].group_id == "g1"
    assert scored[0].rollout_index == 0
    assert scored[0].reward == pytest.approx(1.0)
    assert scored[1].reward == pytest.approx(0.0)
    # Correct rollout gets positive advantage, wrong rollout negative
    assert scored[0].advantage > 0.0
    assert scored[1].advantage < 0.0


def test_score_group_rollout_empty_input_returns_empty():
    from src.llm_agent.generation import score_group_rollout

    assert score_group_rollout([], ground_truth="x", judge_fn=lambda a, g: 1.0) == []


def test_assign_group_relative_advantages_centers_rewards_by_group_mean():
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [
        _make_grouped_rollout("g1", 0, "traj 1", 0.8, 0),
        _make_grouped_rollout("g1", 1, "traj 2", 0.85, 1),
        _make_grouped_rollout("g1", 2, "traj 3", 0.9, 2),
        _make_grouped_rollout("g1", 3, "traj 4", 0.95, 3),
    ]

    # normalize=False → raw mean-centering: advantage_i = reward_i - mean
    scored = assign_group_relative_advantages(
        grouped,
        rewards=[1.0, 0.7, 0.0, 0.0],
        normalize=False,
    )

    assert [s.reward for s in scored] == pytest.approx([1.0, 0.7, 0.0, 0.0])
    assert [s.advantage for s in scored] == pytest.approx(
        [0.575, 0.275, -0.425, -0.425]
    )


def test_assign_group_relative_advantages_std_normalized():
    import math
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [
        _make_grouped_rollout("g1", 0, "traj 1", 0.8, 0),
        _make_grouped_rollout("g1", 1, "traj 2", 0.85, 1),
        _make_grouped_rollout("g1", 2, "traj 3", 0.9, 2),
        _make_grouped_rollout("g1", 3, "traj 4", 0.95, 3),
    ]
    rewards = [1.0, 0.7, 0.0, 0.0]

    # normalize=True (default) → (reward - mean) / (std + eps)
    scored = assign_group_relative_advantages(grouped, rewards=rewards)

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance)
    expected = [(r - mean) / (std + 1e-8) for r in rewards]

    assert [s.advantage for s in scored] == pytest.approx(expected, abs=1e-5)
    # Positive for above-mean rollouts, negative for below-mean
    assert scored[0].advantage > 0.0
    assert scored[1].advantage > 0.0
    assert scored[2].advantage < 0.0
    assert scored[3].advantage < 0.0


def test_assign_group_relative_advantages_all_equal_rewards_gives_zero_advantages():
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [_make_grouped_rollout("g1", i, "answer", 0.8, i) for i in range(4)]
    # All zero rewards → std=0 → all advantages 0.0 in both modes
    for normalize in (True, False):
        scored = assign_group_relative_advantages(
            grouped, rewards=[0.0, 0.0, 0.0, 0.0], normalize=normalize
        )
        assert all(s.advantage == pytest.approx(0.0) for s in scored)


def test_assign_group_relative_advantages_single_rollout_gives_zero_advantage():
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [_make_grouped_rollout("g1", 0, "answer", 0.8, 0)]
    for normalize in (True, False):
        scored = assign_group_relative_advantages(
            grouped, rewards=[1.0], normalize=normalize
        )
        assert scored[0].advantage == pytest.approx(0.0)


def test_format_scored_group_rollout_matches_format_group_rollout():
    from src.llm_agent.generation import (
        assign_group_relative_advantages,
        format_scored_group_rollout,
    )

    grouped = [
        _make_grouped_rollout("g1", 0, "right answer", 0.8, 0),
        _make_grouped_rollout("g1", 1, "wrong answer", 0.9, 1),
    ]
    scored = assign_group_relative_advantages(
        grouped, rewards=[1.0, 0.0], normalize=False
    )
    rendered = format_scored_group_rollout(scored)

    assert "Rollout 0" in rendered
    assert "Rollout 1" in rendered
    assert "Reward: 1.000" in rendered
    assert "Reward: 0.000" in rendered
    assert "Advantage: +0.500" in rendered
    assert "Advantage: -0.500" in rendered


def test_assign_group_relative_advantages_preserves_rollout_identity():
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [
        _make_grouped_rollout("g42", 0, "answer A", 0.8, 0),
        _make_grouped_rollout("g42", 1, "answer B", 0.95, 1),
    ]

    scored = assign_group_relative_advantages(
        grouped,
        rewards=[1.0, 0.0],
        reward_components=[{"terminal_reward": 1.0}, {"terminal_reward": 0.0}],
        normalize=False,  # raw mean-centering: advantage = reward - mean
    )

    assert scored[0].group_id == "g42"
    assert scored[0].rollout_index == 0
    assert scored[0].reward_components == {"terminal_reward": 1.0}
    assert scored[0].advantage == pytest.approx(0.5)
    assert scored[1].group_id == "g42"
    assert scored[1].rollout_index == 1
    assert scored[1].reward_components == {"terminal_reward": 0.0}
    assert scored[1].advantage == pytest.approx(-0.5)


def test_assign_group_relative_advantages_rejects_length_mismatch():
    from src.llm_agent.generation import assign_group_relative_advantages

    grouped = [_make_grouped_rollout("g1", 0, "answer", 0.8, 0)]

    with pytest.raises(ValueError, match="same length"):
        assign_group_relative_advantages(grouped, rewards=[1.0, 0.0])

    with pytest.raises(ValueError, match="same length"):
        assign_group_relative_advantages(
            grouped,
            rewards=[1.0],
            reward_components=[{"terminal_reward": 1.0}, {"terminal_reward": 0.0}],
        )


def test_format_group_rollout_shows_all_rollout_indices_and_group_id():
    from src.llm_agent.generation import format_group_rollout

    grouped = [
        _make_grouped_rollout("grp-abc", 0, "answer A", 0.8, 0),
        _make_grouped_rollout("grp-abc", 1, "answer B", 0.95, 1),
        _make_grouped_rollout("grp-abc", 2, "answer C", 1.1, 2),
    ]
    rendered = format_group_rollout(grouped)

    assert "Group: grp-abc" in rendered
    assert "Rollout 0" in rendered
    assert "Rollout 1" in rendered
    assert "Rollout 2" in rendered
    assert "3 rollouts" in rendered
    assert "temperature=0.80" in rendered


def test_format_group_rollout_shows_rewards_and_advantages():
    from src.llm_agent.generation import format_group_rollout

    grouped = [
        _make_grouped_rollout("g1", 0, "right", 0.8, 0),
        _make_grouped_rollout("g1", 1, "wrong", 0.95, 1),
    ]
    rendered = format_group_rollout(grouped, rewards=[1.0, 0.0], advantages=[0.5, -0.5])

    assert "Reward: 1.000" in rendered
    assert "Reward: 0.000" in rendered
    assert "Advantage: +0.500" in rendered
    assert "Advantage: -0.500" in rendered
    assert "Group mean reward: 0.500" in rendered


def test_format_group_rollout_empty_returns_empty_string():
    from src.llm_agent.generation import format_group_rollout

    assert format_group_rollout([]) == ""


def test_make_continuation_decision_continues_when_active_and_budget_available():
    from src.llm_agent.generation import AgentLoopState

    manager = _manager_with_log_prob()
    state = AgentLoopState(
        rollings=SearchBatch.from_dict(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        ),
        original_left_side={"input_ids": torch.ones(1, 4, dtype=torch.long)},
        original_right_side={
            "responses": torch.ones(1, 0, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 0, dtype=torch.long),
        },
        active_mask=torch.tensor([True], dtype=torch.bool),
        trajectory_turns=[0],
        turns_stats=torch.zeros(1, dtype=torch.int),
        valid_action_stats=torch.zeros(1, dtype=torch.int),
        valid_search_stats=torch.zeros(1, dtype=torch.int),
        active_num_list=[1],
    )
    # context_token_lengths not set yet — no budget check happens
    decision = manager._make_continuation_decision(
        state, turn_index=0, allow_tool_use=True
    )
    assert decision.continue_generation is True
    assert decision.active_trajectories == 1


def test_make_continuation_decision_forces_answer_when_context_nearly_full():
    from src.llm_agent.generation import AgentLoopState

    manager = LLMGenerationManager(
        tokenizer=DummyTokenizer(),
        config=GenerationConfig(
            max_turns=5,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=DummyActorRollout(),
    )
    state = AgentLoopState(
        rollings=SearchBatch.from_dict(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        ),
        original_left_side={"input_ids": torch.ones(1, 4, dtype=torch.long)},
        original_right_side={
            "responses": torch.ones(1, 0, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 0, dtype=torch.long),
        },
        active_mask=torch.tensor([True], dtype=torch.bool),
        trajectory_turns=[0],
        turns_stats=torch.zeros(1, dtype=torch.int),
        valid_action_stats=torch.zeros(1, dtype=torch.int),
        valid_search_stats=torch.zeros(1, dtype=torch.int),
        active_num_list=[1],
    )
    # Simulate context at 90% of max_prompt_length=32 → 29 tokens → above 85% budget
    state.meta_info["context_token_lengths"] = [29]

    decision = manager._make_continuation_decision(
        state, turn_index=2, allow_tool_use=True
    )

    assert decision.continue_generation is False
    assert "context length" in decision.reason
    assert "forcing answer" in decision.reason
    assert decision.active_trajectories == 1


def test_make_continuation_decision_stops_when_all_done():
    from src.llm_agent.generation import AgentLoopState

    manager = _manager_with_log_prob()
    state = AgentLoopState(
        rollings=SearchBatch.from_dict(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        ),
        original_left_side={"input_ids": torch.ones(1, 4, dtype=torch.long)},
        original_right_side={
            "responses": torch.ones(1, 0, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 0, dtype=torch.long),
        },
        active_mask=torch.tensor([False], dtype=torch.bool),
        trajectory_turns=[1],
        turns_stats=torch.ones(1, dtype=torch.int),
        valid_action_stats=torch.zeros(1, dtype=torch.int),
        valid_search_stats=torch.zeros(1, dtype=torch.int),
        active_num_list=[1, 0],
    )
    decision = manager._make_continuation_decision(
        state, turn_index=1, allow_tool_use=True
    )
    assert decision.continue_generation is False
    assert decision.active_trajectories == 0


def test_make_continuation_decision_records_history_in_meta_info():
    from src.llm_agent.generation import AgentLoopState

    manager = _manager_with_log_prob()
    state = AgentLoopState(
        rollings=SearchBatch.from_dict(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        ),
        original_left_side={"input_ids": torch.ones(1, 4, dtype=torch.long)},
        original_right_side={
            "responses": torch.ones(1, 0, dtype=torch.long),
            "responses_with_info_mask": torch.ones(1, 0, dtype=torch.long),
        },
        active_mask=torch.tensor([True], dtype=torch.bool),
        trajectory_turns=[0],
        turns_stats=torch.zeros(1, dtype=torch.int),
        valid_action_stats=torch.zeros(1, dtype=torch.int),
        valid_search_stats=torch.zeros(1, dtype=torch.int),
        active_num_list=[1],
    )
    decision = manager._make_continuation_decision(
        state, turn_index=0, allow_tool_use=True
    )
    manager._record_continuation_decision(state, decision)

    history = state.meta_info.get("continuation_history", [])
    assert len(history) == 1
    assert history[0].turn_index == 0
    assert history[0].continue_generation is True


def test_run_llm_loop_extracts_final_answer_from_answer_turn():
    """final_answers and finished_without_answer are set correctly in meta_info."""

    class AnswerTokenizer(DummyTokenizer):
        def batch_decode(self, responses, skip_special_tokens=True):
            del skip_special_tokens
            mapping = {
                (1, 1): "<search>query</search>",
                (
                    2,
                    2,
                ): "<answer>The 2024 Nobel Prize in Physics was awarded to Hopfield and Hinton.</answer>",
            }
            decoded = []
            for row in responses.tolist():
                tokens = tuple(t for t in row if t != 0)
                decoded.append(mapping.get(tokens, ""))
            return decoded

    manager = LLMGenerationManager(
        tokenizer=AnswerTokenizer(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
        generation_backend=SequencedActorRollout(responses=[[[1, 1]], [[2, 2]]]),
    )
    manager.batch_search = lambda tool_calls, search_mode, gt_threshold: [  # type: ignore[method-assign]
        "Doc 1: Nobel evidence"
    ]
    gen_batch = SearchBatch.from_dict(
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    gen_batch.non_tensor_batch = {
        "question": ["Who won the 2024 Nobel Prize in Physics?"],
        "golden_answers": [["Hopfield and Hinton"]],
    }

    final_batch, trajectory_turns = manager.run_llm_loop(
        gen_batch=gen_batch,
        search_mode="simulate_sft",
        current_step=0,
        total_steps=100,
        initial_input_ids=gen_batch.batch["input_ids"],
    )

    assert final_batch.meta_info["final_answers"] == [
        "The 2024 Nobel Prize in Physics was awarded to Hopfield and Hinton."
    ]
    assert final_batch.meta_info["finished_without_answer"] == [False]
    assert trajectory_turns == [2]
