"""Unit tests for OpenAPI-backed tool providers."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.tools.api import (
    ApiToolError,
    ApiToolNotFoundError,
    ApiToolRegistry,
    parse_openapi_schema,
)


def _schema() -> str:
    return json.dumps(
        {
            "openapi": "3.0.0",
            "info": {"title": "Demo", "description": "Demo API"},
            "servers": [{"url": "https://api.example.test"}],
            "paths": {
                "/suggest/{doctype}": {
                    "get": {
                        "operationId": "YoudaoSuggest",
                        "description": "Return suggestions.",
                        "parameters": [
                            {
                                "name": "doctype",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "q",
                                "in": "query",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "X-Token",
                                "in": "header",
                                "schema": {"type": "string"},
                            },
                        ],
                    }
                },
                "/items": {
                    "post": {
                        "operationId": "createItem",
                        "summary": "Create item.",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "rank": {"type": "integer"},
                                        },
                                        "required": ["name"],
                                    }
                                }
                            }
                        },
                    }
                },
            },
        }
    )


class _FakeResponse:
    headers = {"content-type": "application/json"}

    def __init__(self, request_log):
        self._request_log = request_log

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return {"ok": True, "request": self._request_log[-1]}

    async def text(self):
        return "ok"


class _FakeSession:
    def __init__(self, *, timeout, request_log):
        del timeout
        self._request_log = request_log

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, **kwargs):
        self._request_log.append(kwargs)
        return _FakeResponse(self._request_log)


def test_parse_openapi_schema_validates_json_object():
    parsed = parse_openapi_schema(_schema())

    assert parsed.title == "Demo"
    assert parsed.description == "Demo API"
    assert parsed.server == "https://api.example.test"

    with pytest.raises(ApiToolError):
        parse_openapi_schema("[]")


def test_api_tool_registry_creates_provider_and_tools():
    registry = ApiToolRegistry()

    provider = registry.create_provider(
        name="demo",
        openapi_schema=_schema(),
        headers={"Authorization": "Bearer token"},
    )

    assert provider.description == "Demo API"
    assert sorted(provider.tools) == ["YoudaoSuggest", "createItem"]
    assert provider.tools["YoudaoSuggest"].parameters["required"] == ["doctype", "q"]
    assert "rank" in provider.tools["createItem"].parameters["properties"]
    assert registry.list_providers("dem") == [provider]


def test_api_tool_registry_rejects_duplicate_provider_and_tool_names():
    registry = ApiToolRegistry()
    registry.create_provider(name="demo", openapi_schema=_schema())

    with pytest.raises(ApiToolError, match="already exists"):
        registry.create_provider(name="demo", openapi_schema=_schema())

    duplicate_tool_schema = json.dumps(
        {
            "paths": {
                "/a": {"get": {"operationId": "same"}},
                "/b": {"get": {"operationId": "same"}},
            }
        }
    )
    with pytest.raises(ApiToolError, match="Duplicate"):
        registry.create_provider(name="other", openapi_schema=duplicate_tool_schema)


def test_api_tool_registry_update_rebuilds_tools():
    registry = ApiToolRegistry()
    provider = registry.create_provider(name="demo", openapi_schema=_schema())
    updated_schema = json.dumps(
        {
            "info": {"title": "New"},
            "servers": [{"url": "https://new.example.test"}],
            "paths": {"/ping": {"get": {"operationId": "ping"}}},
        }
    )

    updated = registry.update_provider(
        provider.id,
        name="renamed",
        openapi_schema=updated_schema,
    )

    assert updated.name == "renamed"
    assert sorted(updated.tools) == ["ping"]
    with pytest.raises(ApiToolNotFoundError):
        registry.get_tool(provider.id, "YoudaoSuggest")


def test_api_request_tool_invokes_operation(monkeypatch):
    registry = ApiToolRegistry()
    provider = registry.create_provider(
        name="demo",
        openapi_schema=_schema(),
        headers={"Authorization": "Bearer token"},
    )
    request_log = []

    def _session_factory(*, timeout):
        return _FakeSession(timeout=timeout, request_log=request_log)

    monkeypatch.setattr(
        "src.tools.api.aiohttp.ClientSession",
        _session_factory,
    )

    tool = next(
        tool
        for tool in registry.build_tools(provider.id)
        if tool.name == "YoudaoSuggest"
    )

    text, raw, meta = asyncio.run(
        tool.execute(
            "default",
            {"doctype": "json", "q": "love", "X-Token": "override"},
        )
    )

    request = request_log[0]
    assert request["method"] == "GET"
    assert request["url"] == "https://api.example.test/suggest/json"
    assert request["params"] == {"q": "love"}
    assert request["headers"]["Authorization"] == "Bearer token"
    assert request["headers"]["X-Token"] == "override"
    assert raw["ok"] is True
    assert '"ok": true' in text
    assert meta == {"provider_id": str(provider.id)}


def test_api_request_tool_posts_json_body(monkeypatch):
    registry = ApiToolRegistry()
    provider = registry.create_provider(name="demo", openapi_schema=_schema())
    request_log = []

    def _session_factory(*, timeout):
        return _FakeSession(timeout=timeout, request_log=request_log)

    monkeypatch.setattr(
        "src.tools.api.aiohttp.ClientSession",
        _session_factory,
    )

    asyncio.run(
        registry.invoke_tool(
            provider.id,
            "createItem",
            {"name": "chair", "rank": 2, "ignored": True},
        )
    )

    assert request_log[0]["method"] == "POST"
    assert request_log[0]["url"] == "https://api.example.test/items"
    assert request_log[0]["json"] == {"name": "chair", "rank": 2}
