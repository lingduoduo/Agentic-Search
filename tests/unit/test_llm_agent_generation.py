"""Unit tests for src.llm_agent.generation."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)

from src.agent_loop import build_prompt_dataloader  # noqa: E402
from src.llm_agent.generation import (  # noqa: E402
    AgentLoopState,
    GenerationConfig,
    LLMGenerationManager,
    PPOPolicyLossConfig,
    RolloutTrajectory,
    SearchBatch,
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
    assert batch.non_tensor_batch["trajectories"][0].new_log_probs == [-0.5, -0.5, -0.0]


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

    outputs = manager.build_rollout_outputs(batch)

    assert outputs[0].metrics["repeated_search_queries"] == pytest.approx(
        1.0
    )  # "cats" repeated once
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
