from __future__ import annotations

import io
import json
import urllib.error

import pytest
from pydantic import BaseModel

from src.hooks import HookConfig
from src.hooks import HookExecutionError
from src.hooks import HookFailStrategy
from src.hooks import HookPoint
from src.hooks import HookRegistry
from src.hooks import HookSkipped
from src.hooks import HookSoftFailed
from src.hooks import execute_hook


class QueryHookResponse(BaseModel):
    query: str


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_execute_hook_skips_when_no_hook_is_registered():
    result = execute_hook(
        hook_point=HookPoint.QUERY_PROCESSING,
        payload={"query": "hello"},
        registry=HookRegistry(),
    )

    assert isinstance(result, HookSkipped)


def test_execute_hook_posts_payload_and_validates_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.headers["Authorization"]
        return FakeResponse({"query": "rewritten"})

    monkeypatch.setattr("src.hooks.executor.urllib.request.urlopen", fake_urlopen)
    registry = HookRegistry(
        [
            HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
                api_key="secret",
                timeout_seconds=2,
            )
        ]
    )

    result = execute_hook(
        hook_point=HookPoint.QUERY_PROCESSING,
        payload={"query": "original"},
        response_type=QueryHookResponse,
        registry=registry,
    )

    assert isinstance(result, QueryHookResponse)
    assert result.query == "rewritten"
    assert captured == {
        "url": "https://hooks.test/query",
        "timeout": 2,
        "body": {"query": "original"},
        "auth": "Bearer secret",
    }
    assert registry.execution_log[-1].is_success is True
    assert registry.execution_log[-1].is_reachable is True


def test_execute_hook_soft_failure_returns_marker(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError(
            url="https://hooks.test/query",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr("src.hooks.executor.urllib.request.urlopen", fake_urlopen)

    result = execute_hook(
        hook_point=HookPoint.QUERY_PROCESSING,
        payload={"query": "original"},
        hook=HookConfig(
            hook_point=HookPoint.QUERY_PROCESSING,
            endpoint_url="https://hooks.test/query",
            fail_strategy=HookFailStrategy.SOFT,
        ),
    )

    assert isinstance(result, HookSoftFailed)
    assert result.status_code == 401
    assert result.is_reachable is False


def test_execute_hook_hard_failure_raises(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.URLError("dns failed")

    monkeypatch.setattr("src.hooks.executor.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HookExecutionError, match="unreachable"):
        execute_hook(
            hook_point=HookPoint.QUERY_PROCESSING,
            payload={"query": "original"},
            hook=HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
                fail_strategy=HookFailStrategy.HARD,
            ),
        )
