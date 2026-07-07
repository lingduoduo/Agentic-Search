"""Small helpers for sanitized, bounded runtime feedback text."""

from __future__ import annotations

import re

DEFAULT_HEAD_LINES = 5
DEFAULT_TAIL_LINES = 30
DEFAULT_MAX_CHARS = 4000

_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?key|token)\s*[:=]\s*\S+"
        ),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+"), "xox-[REDACTED]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"
        ),
        "[REDACTED PRIVATE KEY]",
    ),
)


def redact_text(text: str) -> tuple[str, int]:
    """Return text with common secret shapes redacted and the replacement count."""
    if not text:
        return text, 0

    redacted = text
    hits = 0
    for pattern, replacement in _REDACTION_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        hits += count
    return redacted, hits


def _capture_lines(text: str, head_lines: int, tail_lines: int) -> tuple[str, int]:
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return text, 0

    truncated_lines = len(lines) - head_lines - tail_lines
    captured = lines[:head_lines]
    captured.append(f"...truncated {truncated_lines} lines...")
    captured.extend(lines[-tail_lines:])
    return "\n".join(captured), truncated_lines


def deterministic_capture(
    text: str,
    *,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[str, dict[str, int]]:
    """Bound and redact text before persistence.

    The line window preserves the beginning and end of long text. The character
    cap is a final guard for single-line payloads.
    """
    captured, truncated_lines = _capture_lines(text, head_lines, tail_lines)
    redacted, redactions = redact_text(captured)

    truncated_chars = 0
    if max_chars > 0 and len(redacted) > max_chars:
        truncated_chars = len(redacted) - max_chars
        redacted = redacted[:max_chars]

    return redacted, {
        "truncated_lines": truncated_lines,
        "truncated_chars": truncated_chars,
        "redactions": redactions,
    }
