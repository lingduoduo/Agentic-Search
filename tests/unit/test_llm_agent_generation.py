"""Unit tests for src.llm_agent.generation."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)

from src.llm_agent.generation import GenerationConfig, LLMGenerationManager, SearchBatch  # noqa: E402


class DummyTokenizer:
    pad_token_id = 0

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
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action_tag == "search"
    assert final_batch.batch["responses"].shape[1] > 0


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
