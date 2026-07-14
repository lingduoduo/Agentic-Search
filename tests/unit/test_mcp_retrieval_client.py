"""Tests for authenticated retrieval through the web backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastmcp.server.auth.auth import AccessToken

from src.internal.mcp_server.retrieval_client import AuthenticatedDocument
from src.internal.mcp_server.retrieval_client import AuthenticatedRetrievalError
from src.internal.mcp_server.retrieval_client import authenticated_retrieve


def _token() -> AccessToken:
    return AccessToken(token="secret", client_id="mcp", scopes=[])


def _response(status_code: int = 200, payload: object | None = None) -> Mock:
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "request failed",
            request=httpx.Request("POST", "http://web/search/send-search-message"),
            response=httpx.Response(status_code),
        )
    return response


@pytest.mark.asyncio
async def test_forwards_token_and_normalizes_payload() -> None:
    response = _response(payload={"all_executed_queries": ["GRPO"], "search_docs": []})
    client = Mock()
    client.post = AsyncMock(return_value=response)

    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.build_web_base_url",
            return_value="http://web",
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
    ):
        result = await authenticated_retrieve(
            "GRPO", top_k=5, document_set_names=["ml"]
        )

    assert result == []
    client.post.assert_awaited_once_with(
        "http://web/search/send-search-message",
        headers={"Authorization": "Bearer secret"},
        json={
            "search_query": "GRPO",
            "filters": {"document_sets": ["ml"]},
            "run_query_expansion": False,
            "num_hits": 5,
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_normalizes_empty_document_sets_to_no_filters() -> None:
    response = _response(payload={"all_executed_queries": ["q"], "search_docs": []})
    client = Mock(post=AsyncMock(return_value=response))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.build_web_base_url",
            return_value="http://web",
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
    ):
        await authenticated_retrieve("q", top_k=3, document_set_names=[])

    assert client.post.await_args.kwargs["json"]["filters"] is None


@pytest.mark.asyncio
async def test_parses_successful_response() -> None:
    payload = {
        "all_executed_queries": ["q"],
        "search_docs": [
            {
                "title": "A",
                "url": None,
                "content": "body",
                "score": 0.75,
                "metadata": {"source": "wiki"},
            }
        ],
    }
    client = Mock(post=AsyncMock(return_value=_response(payload=payload)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
    ):
        result = await authenticated_retrieve("q", top_k=1)

    assert result == [
        AuthenticatedDocument(
            title="A", url=None, content="body", score=0.75, metadata={"source": "wiki"}
        )
    ]


@pytest.mark.asyncio
async def test_translates_backend_error_in_success_response_without_leaking_it() -> (
    None
):
    payload = {
        "all_executed_queries": ["q"],
        "search_docs": [],
        "error": "database password secret-db-password",
    }
    client = Mock(post=AsyncMock(return_value=_response(payload=payload)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
        pytest.raises(
            AuthenticatedRetrievalError, match="Search request failed"
        ) as exc,
    ):
        await authenticated_retrieve("q", top_k=1)

    assert "secret-db-password" not in str(exc.value)


@pytest.mark.asyncio
async def test_normalizes_null_title_to_empty_string() -> None:
    payload = {
        "all_executed_queries": ["q"],
        "search_docs": [
            {
                "title": None,
                "url": "https://example.com",
                "content": "body",
                "score": 0.5,
                "metadata": {},
            }
        ],
    }
    client = Mock(post=AsyncMock(return_value=_response(payload=payload)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
    ):
        result = await authenticated_retrieve("q", top_k=1)

    assert result[0].title == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message"),
    [(401, "Authentication failed"), (403, "Access to search results was denied")],
)
async def test_translates_auth_errors(status_code: int, message: str) -> None:
    client = Mock(post=AsyncMock(return_value=_response(status_code)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
        pytest.raises(AuthenticatedRetrievalError, match=message),
    ):
        await authenticated_retrieve("q", top_k=1)


@pytest.mark.asyncio
async def test_translates_server_errors_without_leaking_response() -> None:
    client = Mock(post=AsyncMock(return_value=_response(503)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
        pytest.raises(
            AuthenticatedRetrievalError, match="Search service is unavailable"
        ),
    ):
        await authenticated_retrieve("q", top_k=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.TimeoutException("secret timeout"), httpx.ConnectError("secret host")],
)
async def test_translates_transport_errors(error: Exception) -> None:
    client = Mock(post=AsyncMock(side_effect=error))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
        pytest.raises(
            AuthenticatedRetrievalError, match="Could not reach the search service"
        ) as exc,
    ):
        await authenticated_retrieve("q", top_k=1)
    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"search_docs": "invalid"},
        {"search_docs": [{"title": "missing fields"}]},
    ],
)
async def test_rejects_malformed_responses(payload: object) -> None:
    client = Mock(post=AsyncMock(return_value=_response(payload=payload)))
    with (
        patch(
            "src.internal.mcp_server.retrieval_client.require_access_token",
            return_value=_token(),
        ),
        patch(
            "src.internal.mcp_server.retrieval_client.get_http_client",
            return_value=client,
        ),
        pytest.raises(AuthenticatedRetrievalError, match="invalid response"),
    ):
        await authenticated_retrieve("q", top_k=1)


def test_does_not_depend_on_raw_retrieval_helper() -> None:
    import src.internal.mcp_server.retrieval_client as retrieval_client

    assert not hasattr(retrieval_client, "retrieval_search")
