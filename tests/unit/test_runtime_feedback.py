"""Tests for runtime feedback sanitization helpers."""

from __future__ import annotations

from src.internal.feedback.runtime import deterministic_capture, redact_text


def test_redact_text_masks_common_secret_shapes():
    text = "Authorization: Bearer abc.def\na password=hunter2\nAKIAIOSFODNN7EXAMPLE"
    redacted, hits = redact_text(text)
    assert hits == 3
    assert "hunter2" not in redacted
    assert "abc.def" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted


def test_deterministic_capture_keeps_head_tail_and_counts_removed_lines():
    text = "\n".join(f"line {i}" for i in range(10))
    captured, meta = deterministic_capture(
        text, head_lines=2, tail_lines=2, max_chars=1000
    )
    assert captured.splitlines() == [
        "line 0",
        "line 1",
        "...truncated 6 lines...",
        "line 8",
        "line 9",
    ]
    assert meta["truncated_lines"] == 6
    assert meta["redactions"] == 0


def test_deterministic_capture_redacts_before_returning_text():
    captured, meta = deterministic_capture("token=secret-value", max_chars=1000)
    assert "secret-value" not in captured
    assert meta["redactions"] == 1
