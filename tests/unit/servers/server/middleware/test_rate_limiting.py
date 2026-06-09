"""Tests for src/backend/servers/middleware/rate_limiting.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.backend.servers.middleware.rate_limiting import (
    _make_key,
    _make_limiter,
    _windows,
)


def _request(ip: str = "1.2.3.4", ua: str = "test-agent") -> MagicMock:
    req = MagicMock()
    req.client.host = ip
    req.headers.get = lambda k, default="": ua if k == "user-agent" else default
    return req


@pytest.fixture(autouse=True)
def clear_windows():
    _windows.clear()
    yield
    _windows.clear()


class TestMakeKey:
    def test_combines_ip_and_ua(self):
        key = _make_key(_request("10.0.0.1", "Mozilla/5.0"))
        assert key == "10.0.0.1-Mozilla/5.0"

    def test_spaces_in_ua_replaced(self):
        key = _make_key(_request("10.0.0.1", "My Agent 1.0"))
        assert " " not in key

    def test_no_client_falls_back(self):
        req = MagicMock()
        req.client = None
        req.headers.get = lambda k, default="": "agent"
        key = _make_key(req)
        assert key.startswith("unknown-")


class TestMakeLimiter:
    async def test_allows_requests_under_limit(self):
        limiter = _make_limiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            await limiter(_request())

    async def test_blocks_on_limit_exceeded(self):
        limiter = _make_limiter(max_requests=2, window_seconds=60)
        await limiter(_request())
        await limiter(_request())
        with pytest.raises(HTTPException) as exc_info:
            await limiter(_request())
        assert exc_info.value.status_code == 429

    async def test_different_clients_tracked_separately(self):
        limiter = _make_limiter(max_requests=1, window_seconds=60)
        await limiter(_request(ip="1.1.1.1"))
        # different IP should not be affected
        await limiter(_request(ip="2.2.2.2"))

    async def test_expired_timestamps_not_counted(self):
        limiter = _make_limiter(max_requests=2, window_seconds=1)
        key = _make_key(_request())
        # plant a timestamp already outside the window
        _windows[key] = [time.monotonic() - 2]
        # should allow two more requests despite the planted entry
        await limiter(_request())
        await limiter(_request())

    async def test_window_boundary_exact(self):
        limiter = _make_limiter(max_requests=1, window_seconds=60)
        await limiter(_request())
        with pytest.raises(HTTPException):
            await limiter(_request())


class TestGetAuthRateLimiters:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AUTH_RATE_LIMITING_ENABLED", raising=False)
        # reload to pick up env change
        import importlib
        import src.backend.servers.middleware.rate_limiting as mod

        importlib.reload(mod)
        assert mod.get_auth_rate_limiters() == []

    def test_enabled_returns_one_dependency(self, monkeypatch):
        monkeypatch.setenv("AUTH_RATE_LIMITING_ENABLED", "true")
        import importlib
        import src.backend.servers.middleware.rate_limiting as mod

        importlib.reload(mod)
        limiters = mod.get_auth_rate_limiters()
        assert len(limiters) == 1
