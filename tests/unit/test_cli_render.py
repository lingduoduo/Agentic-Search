# tests/unit/test_cli_render.py
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from rich.console import Console

from src.cli._render import render_answer_progressive, render_sources

DOCS = [
    {
        "citation": "[1]",
        "title": "Q3 Financial Report",
        "url": "https://internal.corp/reports/q3",
        "content": "x",
    },
    {"citation": "[2]", "title": None, "url": None, "content": "y"},
]


def test_render_sources_prints_table(capsys):
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False)
    with patch("src.cli._render.console", test_console):
        render_sources(DOCS)
    output = buf.getvalue()
    assert "Q3 Financial Report" in output
    assert "internal.corp" in output
    assert "[1]" in output


def test_render_sources_empty_list_prints_nothing(capsys):
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False)
    with patch("src.cli._render.console", test_console):
        render_sources([])
    assert buf.getvalue() == ""


def test_render_answer_progressive_calls_live_update():
    live_mock = MagicMock()
    live_mock.__enter__ = MagicMock(return_value=live_mock)
    live_mock.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.cli._render.Live", return_value=live_mock),
        patch("src.cli._render.time.sleep"),
    ):
        render_answer_progressive("Hello world test", words_per_second=1000.0)

    # Live.update called once per word (3 words)
    assert live_mock.update.call_count == 3


def test_render_answer_progressive_empty_string():
    live_mock = MagicMock()
    live_mock.__enter__ = MagicMock(return_value=live_mock)
    live_mock.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.cli._render.Live", return_value=live_mock),
        patch("src.cli._render.time.sleep"),
    ):
        render_answer_progressive("", words_per_second=1000.0)

    assert live_mock.update.call_count == 0
