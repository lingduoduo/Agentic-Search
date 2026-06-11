"""Tests for license API utilities (src/servers/license/api.py)."""

from __future__ import annotations

from src.internal.servers.license.api import _strip_pem

# The project uses "AGENTIC SEARCH LICENSE" as the PEM header.
_BEGIN = "-----BEGIN AGENTIC SEARCH LICENSE-----"
_END = "-----END AGENTIC SEARCH LICENSE-----"


class TestStripPem:
    """Tests for the PEM delimiter stripping helper."""

    def test_strips_pem_delimiters(self) -> None:
        content = f"{_BEGIN}\neyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ==\n{_END}"
        assert _strip_pem(content) == "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="

    def test_handles_multiline_content(self) -> None:
        body = "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjog\nIjEuMCJ9fQ=="
        content = f"{_BEGIN}\n{body}\n{_END}"
        assert _strip_pem(content) == body

    def test_returns_unchanged_without_delimiters(self) -> None:
        raw = "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="
        assert _strip_pem(raw) == raw

    def test_handles_surrounding_whitespace(self) -> None:
        content = (
            f"\n  {_BEGIN}\neyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ==\n{_END}\n  "
        )
        assert _strip_pem(content) == "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="

    def test_begin_only_returns_stripped_input(self) -> None:
        content = f"{_BEGIN}\neyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="
        result = _strip_pem(content)
        # Without both delimiters the function returns the stripped input unchanged
        assert "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ==" in result

    def test_trailing_newlines_stripped_from_raw(self) -> None:
        content = "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ==\n\n"
        assert _strip_pem(content) == "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="

    def test_trailing_newline_inside_pem_stripped(self) -> None:
        content = f"{_BEGIN}\neyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ==\n\n{_END}"
        assert _strip_pem(content) == "eyJwYXlsb2FkIjogeyJ2ZXJzaW9uIjogIjEuMCJ9fQ=="
