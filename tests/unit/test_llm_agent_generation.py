"""Unit tests for src.llm_agent.generation."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)

from src.llm_agent.generation import GenerationConfig, LLMGenerationManager


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False, return_tensors="pt", padding="longest"):
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


def _manager() -> LLMGenerationManager:
    return LLMGenerationManager(
        tokenizer=DummyTokenizer(),
        actor_rollout_wg=DummyActorRollout(),
        config=GenerationConfig(
            max_turns=2,
            max_start_length=8,
            max_prompt_length=32,
            max_response_length=16,
            max_obs_length=16,
            num_gpus=1,
        ),
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
