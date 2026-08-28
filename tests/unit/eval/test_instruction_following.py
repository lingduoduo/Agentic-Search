"""Contracts for the four objective instruction-following predicates.

Objective on purpose: no judge, so the headline number does not depend on a
third model and does not move when that model is swapped.
"""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import EvalRecord
from src.model.post_training.eval.instruction_following import (
    CONSTRAINT_NAMES,
    check_constraints,
)

TOOLS = frozenset({"search", "fetch"})


def _record(**overrides) -> EvalRecord:
    base = {
        "user_id": "u0",
        "prompt_id": "p0",
        "policy": "trained",
        "reward": 0.0,
        "converted": False,
        "response": "<answer>grounded [R1Q1D1]</answer>",
        "metrics": {"rounds_used": 2.0},
        "cited_ids": frozenset({"R1Q1D1"}),
        "tool_calls": ('{"name": "search", "arguments": {}}',),
    }
    base.update(overrides)
    return EvalRecord(**base)


def test_a_fully_compliant_record_passes_everything():
    assert check_constraints(
        _record(), allowed_tools=TOOLS, max_search_rounds=5
    ) == dict.fromkeys(CONSTRAINT_NAMES, True)


def test_result_always_reports_every_constraint():
    result = check_constraints(
        _record(response="", tool_calls=()), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert set(result) == set(CONSTRAINT_NAMES)


@pytest.mark.parametrize(
    "response",
    ["no tag at all", "<answer>unclosed", "answer</answer>", "<answer></answer>"],
)
def test_malformed_answer_tags_fail(response: str):
    result = check_constraints(
        _record(response=response), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["answer_tag_present"] is False


def test_a_citation_label_that_was_never_retrieved_fails():
    """Parsing is not enough -- the label must resolve to a retrieved doc."""
    result = check_constraints(
        _record(response="<answer>x [R9Q9D9]</answer>", cited_ids=frozenset()),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["citations_wellformed"] is False


def test_an_answer_citing_nothing_fails_the_citation_constraint():
    result = check_constraints(
        _record(response="<answer>ungrounded</answer>", cited_ids=frozenset()),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["citations_wellformed"] is False


def test_unparseable_tool_calls_fail():
    result = check_constraints(
        _record(tool_calls=("{not json",)), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["tool_calls_parseable"] is False


def test_valid_json_naming_an_unregistered_tool_fails():
    result = check_constraints(
        _record(tool_calls=('{"name": "rm_rf", "arguments": {}}',)),
        allowed_tools=TOOLS,
        max_search_rounds=5,
    )

    assert result["tool_calls_parseable"] is False


def test_a_record_with_no_tool_calls_vacuously_satisfies_the_constraint():
    result = check_constraints(
        _record(tool_calls=()), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["tool_calls_parseable"] is True


def test_exceeding_the_round_budget_fails():
    result = check_constraints(
        _record(metrics={"rounds_used": 9.0}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is False


def test_exactly_the_budget_is_allowed():
    result = check_constraints(
        _record(metrics={"rounds_used": 5.0}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is True


def test_a_missing_rounds_metric_counts_as_zero_rounds():
    result = check_constraints(
        _record(metrics={}), allowed_tools=TOOLS, max_search_rounds=5
    )

    assert result["round_budget_respected"] is True
