"""Unit tests for the public_data HTTP layer. No live network."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.internal.tools.public_data import _http
from src.internal.tools.public_data._http import (
    PublicDataError,
    get_json,
    guarded,
)


class _FakeResponse:
    def __init__(self, *, status=200, body="{}"):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._body


class _FakeSession:
    """Records the single request it is given, then replays a canned response."""

    calls: list[dict] = []

    def __init__(self, *, status=200, body="{}", raises=None):
        self._status = status
        self._body = body
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        if self._raises is not None:
            raise self._raises
        _FakeSession.calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(status=self._status, body=self._body)


def _install(monkeypatch, **kwargs):
    _FakeSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return _FakeSession(**kwargs)

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)


def test_get_json_parses_body_and_sends_user_agent(monkeypatch):
    _install(monkeypatch, body=json.dumps({"ok": True}))

    result = asyncio.run(get_json("https://example.test/x", params={"a": "b"}))

    assert result == {"ok": True}
    call = _FakeSession.calls[0]
    assert call["method"] == "GET"
    assert call["params"] == {"a": "b"}
    assert call["headers"]["User-Agent"] == _http.USER_AGENT


def test_get_json_caller_headers_override_default(monkeypatch):
    _install(monkeypatch, body="{}")

    asyncio.run(
        get_json("https://example.test/x", headers={"User-Agent": "Mozilla/5.0"})
    )

    assert _FakeSession.calls[0]["headers"]["User-Agent"] == "Mozilla/5.0"


def test_get_json_raises_on_http_error(monkeypatch):
    _install(monkeypatch, status=503, body="down")

    with pytest.raises(PublicDataError) as excinfo:
        asyncio.run(get_json("https://example.test/x"))

    assert "503" in str(excinfo.value)


def test_get_json_raises_on_non_json_body(monkeypatch):
    _install(monkeypatch, body="<html>nope</html>")

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_get_json_raises_on_transport_failure(monkeypatch):
    _install(monkeypatch, raises=asyncio.TimeoutError())

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_guarded_serializes_success():
    @guarded
    async def _ok(value: str):
        return {"value": value}

    assert json.loads(asyncio.run(_ok(value="hi"))) == {"value": "hi"}


def test_guarded_converts_public_data_error():
    @guarded
    async def _boom():
        raise PublicDataError("upstream is down")

    assert json.loads(asyncio.run(_boom())) == {"error": "upstream is down"}


def test_guarded_converts_unexpected_error():
    @guarded
    async def _boom():
        raise KeyError("missing")

    assert "error" in json.loads(asyncio.run(_boom()))


def test_guarded_result_is_a_coroutine_function():
    """FunctionTool.execute awaits only if iscoroutinefunction() is True."""
    import inspect

    @guarded
    async def _ok():
        return {}

    assert inspect.iscoroutinefunction(_ok)
