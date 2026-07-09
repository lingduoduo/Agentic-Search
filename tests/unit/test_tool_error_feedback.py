"""Unit tests for tool-result → tool-message formatting (error feedback)."""

from __future__ import annotations

import json

from src.agents.core.state import (
    PerformanceMetrics,
    TaskStatus,
    ToolExecutionResult,
)
from src.agents.tool.tool_calling import ToolAgentLoop


def _result(status, *, result=None, error_code=None, error_message=None):
    return ToolExecutionResult(
        tool_name="t",
        status=status,
        result=result,
        arguments={},
        performance=PerformanceMetrics(execution_time=0.0, success_rate=0.0),
        error_code=error_code,
        error_message=error_message,
    )


def test_completed_returns_raw_result():
    r = _result(TaskStatus.COMPLETED, result={"a": 1})
    assert ToolAgentLoop._tool_message_content(r) == str({"a": 1})


def test_skipped_format_unchanged():
    r = _result(
        TaskStatus.SKIPPED, error_code="approval_denied", error_message="skipped msg"
    )
    assert ToolAgentLoop._tool_message_content(r) == json.dumps(
        {"status": "skipped", "error_code": "approval_denied"}
    )


def test_failed_includes_error_message():
    r = _result(TaskStatus.FAILED, error_code="ValueError", error_message="nope")
    assert json.loads(ToolAgentLoop._tool_message_content(r)) == {
        "status": "failed",
        "error_code": "ValueError",
        "error_message": "nope",
    }
