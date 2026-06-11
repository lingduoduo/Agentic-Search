"""Unit tests for the Bamboogle evaluation module.

Tests run without any network access — dataset loading is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.training.eval.bamboogle import (
    contains_match,
    evaluate_bamboogle,
    exact_match,
    load_bamboogle,
    normalize_text,
    _extract_answer,
    _make_judge_fn,
    _to_loop_output,
)
from src.agents.base import AgentLoopOutput


# ---------------------------------------------------------------------------
# normalize / exact_match / contains_match
# ---------------------------------------------------------------------------


def test_normalize_strips_articles():
    assert normalize_text("The quick Brown Fox") == "quick brown fox"


def test_normalize_removes_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_exact_match_hit():
    assert exact_match("Albert Einstein", ["Albert Einstein", "Einstein"]) == 1.0


def test_exact_match_miss():
    assert exact_match("Newton", ["Albert Einstein"]) == 0.0


def test_exact_match_normalises():
    assert exact_match("the eiffel tower", ["Eiffel Tower"]) == 1.0


def test_contains_match_substring():
    assert contains_match("The answer is Paris, France.", ["Paris"]) == 1.0


def test_contains_match_miss():
    assert contains_match("Berlin is the answer.", ["Paris"]) == 0.0


def test_contains_match_multiple_gold():
    assert contains_match("Rome wasn't built in a day.", ["Athens", "Rome"]) == 1.0


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


def test_extract_answer_from_string():
    assert _extract_answer("hello") == "hello"


def test_extract_answer_from_dict():
    assert _extract_answer({"answer": "Paris"}) == "Paris"


def test_extract_answer_from_object():
    obj = MagicMock()
    obj.answer = "Berlin"
    assert _extract_answer(obj) == "Berlin"


def test_extract_answer_missing_returns_empty():
    assert _extract_answer({}) == ""


# ---------------------------------------------------------------------------
# _make_judge_fn
# ---------------------------------------------------------------------------


def test_make_judge_fn_match():
    judge = _make_judge_fn(["Paris", "Pairs"])
    assert judge("Paris is the capital", "_ignored_") == 1.0


def test_make_judge_fn_no_match():
    judge = _make_judge_fn(["London"])
    assert judge("Paris is the capital", "_ignored_") == 0.0


# ---------------------------------------------------------------------------
# _to_loop_output
# ---------------------------------------------------------------------------


def test_to_loop_output_stub_from_agent_result():
    obj = MagicMock()
    obj.answer = "some answer"
    del obj.metadata  # no metadata attribute
    out = _to_loop_output(obj)
    assert isinstance(out, AgentLoopOutput)
    assert out.final_answer == "some answer"
    assert out.prompt_ids == []


def test_to_loop_output_rich_path():
    real_output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        final_answer="rich answer",
        metrics={"rounds_used": 1.0},
    )
    obj = MagicMock()
    obj.metadata = {"loop_output": real_output}
    out = _to_loop_output(obj)
    assert out is real_output
    assert out.metrics["rounds_used"] == 1.0


# ---------------------------------------------------------------------------
# load_bamboogle (mocked network)
# ---------------------------------------------------------------------------

_FAKE_JSONL = "\n".join(
    json.dumps(
        {
            "id": str(i),
            "question": f"Q{i}",
            "golden_answers": [f"A{i}"],
        }
    )
    for i in range(5)
)


@patch("requests.get")
def test_load_bamboogle_all(mock_get):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle()
    assert len(rows) == 5


@patch("requests.get")
def test_load_bamboogle_limit(mock_get):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# evaluate_bamboogle (fully mocked)
# ---------------------------------------------------------------------------

_FAKE_DATASET = [
    {
        "id": "1",
        "question": "Who invented the telephone?",
        "golden_answers": ["Alexander Graham Bell"],
    },
    {
        "id": "2",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"],
    },
]


class _PerfectAgent:
    """Returns the first gold answer verbatim."""

    def invoke(self, state: dict) -> MagicMock:
        q = state["messages"][0]["content"]
        answers = {
            "Who invented the telephone?": "Alexander Graham Bell",
            "What is the capital of France?": "Paris",
        }
        result = MagicMock()
        result.answer = answers.get(q, "")
        del result.metadata
        return result


class _WrongAgent:
    def invoke(self, state: dict) -> MagicMock:
        result = MagicMock()
        result.answer = "wrong answer"
        del result.metadata
        return result


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_perfect_agent(mock_load, tmp_path):
    summary, rows = evaluate_bamboogle(
        _PerfectAgent(),
        limit=2,
        output_path=tmp_path / "out.jsonl",
        verbose=False,
    )
    assert summary.exact_match == pytest.approx(1.0)
    assert summary.contains_match == pytest.approx(1.0)
    assert summary.num_examples == 2
    assert summary.avg_reward is None


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_wrong_agent(mock_load, tmp_path):
    summary, rows = evaluate_bamboogle(
        _WrongAgent(),
        limit=2,
        output_path=None,
        verbose=False,
    )
    assert summary.exact_match == pytest.approx(0.0)
    assert summary.contains_match == pytest.approx(0.0)


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_writes_jsonl(mock_load, tmp_path):
    out = tmp_path / "sub" / "out.jsonl"
    evaluate_bamboogle(_PerfectAgent(), limit=2, output_path=out, verbose=False)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert "prediction" in row
    assert "exact_match" in row
    assert "reward_components" in row


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_with_reward_fn(mock_load, tmp_path):
    from src.training.reward import SearchRewardFunction, SearchRewardConfig

    reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    summary, rows = evaluate_bamboogle(
        _PerfectAgent(),
        reward_fn=reward_fn,
        limit=2,
        output_path=None,
        verbose=False,
    )
    assert summary.avg_reward is not None
    assert summary.avg_reward > 0.0  # perfect agent should score non-zero correctness
    for row in rows:
        assert row.reward_total is not None
        assert "total" in row.reward_components
        assert "correctness" in row.reward_components


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_reward_uses_gold_list(mock_load):
    """judge_fn must check against all gold answers, not just the first."""
    from src.training.reward import SearchRewardFunction, SearchRewardConfig

    dataset = [
        {
            "id": "1",
            "question": "Who?",
            "golden_answers": ["Alexander Graham Bell", "Bell"],
        }
    ]
    mock_load.return_value = dataset

    class _SecondGoldAgent:
        def invoke(self, state):
            r = MagicMock()
            r.answer = "Bell"  # matches second gold answer only
            del r.metadata
            return r

    reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    summary, _ = evaluate_bamboogle(
        _SecondGoldAgent(),
        reward_fn=reward_fn,
        limit=1,
        output_path=None,
        verbose=False,
    )
    assert summary.avg_reward > 0.0
