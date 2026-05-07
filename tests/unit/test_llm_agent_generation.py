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


def test_postprocess_predictions_extracts_search_and_answer_tags():
    manager = _manager()
    actions, contents = manager.postprocess_predictions(
        ["before <search>cats</search> after", "<answer>42</answer>", "plain text"]
    )
    assert actions == ["search", "answer", None]
    assert contents == ["cats", "42", ""]


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

    assert captured["payload"] == [("cats", "second question", "second answer")]
    assert captured["search_mode"] == "google"
    assert captured["gt_threshold"] == 0.8
    assert next_obs[1] == "\n\n<information>doc for second question</information>\n\n"
    assert dones == [1, 0]
    assert valid_action == [1, 1]
    assert is_search == [0, 1]


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


def test_search_returns_fallback_for_unknown_mode():
    manager = _manager()
    result, index = manager._search("query", "problem", "answer", "unknown", 0.5, 3)
    assert result == "No information available"
    assert index == 3


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
    assert final_batch.batch["responses"].shape[1] > 0


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
    assert captured["search_mode"] == "google"
