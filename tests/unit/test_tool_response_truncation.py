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


def test_text_under_the_limit_is_returned_unchanged():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = '[{"a": 1}]'
    assert _truncate_tool_text(text, 100, "left") == text


def test_json_array_over_the_limit_keeps_the_leading_items():
    from src.agents.tool.tool_calling import _truncate_tool_text

    items = [{"rank": i, "a": "X" * 100} for i in range(5)]

    result = _truncate_tool_text(json.dumps(items), 400, "left")

    body, _, _footer = result.partition("\n...")
    kept = json.loads(body)
    assert [item["rank"] for item in kept] == list(range(len(kept)))
    assert len(kept) < len(items)


def test_prose_over_the_limit_keeps_the_head_by_default():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "left")

    assert result.startswith("FIRST")
    assert "LAST" not in result
    assert result.endswith("...(truncated)")


def test_right_side_still_keeps_the_tail():
    """The knob keeps working for anyone who sets it explicitly."""
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "right")

    assert result.startswith("(truncated)...")
    assert result.endswith("LAST")


def test_middle_side_keeps_both_ends():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = "FIRST" + "." * 100 + "LAST"

    result = _truncate_tool_text(text, 50, "middle")

    assert result.startswith("FIRST")
    assert result.endswith("LAST")
    assert "...(truncated)..." in result


def test_json_object_over_the_limit_falls_back_to_slicing():
    from src.agents.tool.tool_calling import _truncate_tool_text

    text = json.dumps({"temperature": 14.2, "note": "Y" * 200})

    result = _truncate_tool_text(text, 50, "left")

    assert result.startswith('{"temperature"')
    assert result.endswith("...(truncated)")


def test_the_default_truncation_side_keeps_the_head():
    """A ranked tool result must not lose its top entries."""
    from src.agents.tool.tool_calling import ToolAgentLoopConfig

    assert ToolAgentLoopConfig().tool_response_truncate_side == "left"
