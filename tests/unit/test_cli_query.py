# tests/unit/test_cli_query.py
from __future__ import annotations

from unittest.mock import patch


from src.cli._client import AgentResult
from src.cli.query import main

RESULT = AgentResult(
    session_id="s1",
    answer="Revenue grew 12% year-over-year.",
    citations=["[1]"],
    documents=[
        {
            "citation": "[1]",
            "title": "Q3 Report",
            "url": "https://internal.corp/q3",
            "content": "c",
        }
    ],
)


def test_main_returns_0_on_success():
    with (
        patch("src.cli.query.resolve_token", return_value="tok"),
        patch("src.cli.query.asyncio.run", return_value=RESULT),
        patch("src.cli.query.render_sources"),
        patch("src.cli.query.render_answer_progressive"),
        patch("src.cli.query.console"),
    ):
        code = main(["show me the Q3 report", "--token", "tok", "--url", "http://x"])
    assert code == 0


def test_main_returns_1_on_auth_error():
    with (
        patch("src.cli.query.resolve_token", side_effect=ValueError("no auth")),
        patch("src.cli.query.console"),
    ):
        code = main(["q", "--url", "http://x"])
    assert code == 1


def test_main_returns_1_on_request_error():
    with (
        patch("src.cli.query.resolve_token", return_value="tok"),
        patch("src.cli.query.asyncio.run", side_effect=Exception("connection refused")),
        patch("src.cli.query.console"),
    ):
        code = main(["q", "--token", "tok", "--url", "http://x"])
    assert code == 1


def test_main_returns_1_when_query_empty():
    with patch("src.cli.query.console") as mock_console:
        mock_console.input.return_value = ""
        code = main(["--token", "tok", "--url", "http://x"])
    assert code == 1
