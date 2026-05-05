"""Unit tests for src.agent_loop.search_client."""

from __future__ import annotations

import asyncio

from src.agent_loop.search_client import SearchClient, SearchClientConfig


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload, session_log):
        self._payload = payload
        self._session_log = session_log
        self.closed = False
        self.post_calls = []
        self._session_log.append(self)

    def post(self, url, json):
        self.post_calls.append((url, json))
        return _FakeResponse(self._payload)

    async def close(self):
        self.closed = True


def test_search_client_reuses_session_across_requests(monkeypatch):
    sessions = []

    def _session_factory(*, timeout):
        del timeout
        return _FakeSession(
            {"result": [[{"document": {"title": "T", "contents": '"T"\nBody'}}]]},
            sessions,
        )

    monkeypatch.setattr(
        "src.agent_loop.search_client.aiohttp.ClientSession",
        _session_factory,
    )

    client = SearchClient(SearchClientConfig(url="http://localhost:8000/retrieve"))

    async def _exercise_client():
        first = await client.retrieve(["first"])
        second = await client.retrieve(["second"])
        await client.aclose()
        return first, second

    first, second = asyncio.run(_exercise_client())

    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert len(sessions[0].post_calls) == 2
    assert first[0][0].title == "T"
    assert second[0][0].contents == '"T"\nBody'
