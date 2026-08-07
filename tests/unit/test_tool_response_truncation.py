"""Unit tests for tool-response truncation helpers."""

from __future__ import annotations

import json

from src.agents.tool.tool_calling import _fit_json_array, _truncation_footer


def test_footer_reports_counts():
    assert (
        _truncation_footer(2, 5, 3)
        == "\n...2 of 5 results shown, 3 omitted for length."
    )


def test_returns_none_for_non_json():
    assert _fit_json_array("1. Some Page\nA prose summary.", 10) is None


def test_returns_none_for_malformed_json():
    assert _fit_json_array('[{"a": 1}, {"b":', 10) is None


def test_returns_none_for_json_object():
    assert _fit_json_array('{"temperature": 14.2}', 5) is None


def test_returns_none_for_empty_list():
    assert _fit_json_array("[]", 1) is None


def test_returns_none_when_not_even_one_item_fits():
    """An empty array would be valid JSON but tells the model nothing."""
    text = json.dumps([{"a": "X" * 500}])
    assert _fit_json_array(text, 50) is None


def test_keeps_the_first_items_that_fit():
    items = [{"a": "X" * 100} for _ in range(5)]
    limit = 400

    result = _fit_json_array(json.dumps(items), limit)

    assert result is not None
    assert len(result) <= limit
    body, separator, footer = result.partition("\n...")
    assert separator, "expected a footer when items were dropped"
    kept = json.loads(body)  # must be valid JSON
    assert 1 <= len(kept) < len(items)
    # A prefix, in original order — the top-ranked items, not the tail.
    assert kept == items[: len(kept)]
    assert footer == (
        f"{len(kept)} of {len(items)} results shown, "
        f"{len(items) - len(kept)} omitted for length."
    )


def test_compacting_alone_can_make_everything_fit():
    """Indented input can shrink under the limit once re-serialized."""
    items = [{"a": 1}, {"b": 2}]
    text = json.dumps(items, indent=4)
    limit = len(text) - 1
    assert len(json.dumps(items)) <= limit

    result = _fit_json_array(text, limit)

    assert result == json.dumps(items)
    assert "omitted for length" not in result
