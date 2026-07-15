from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from starlette.requests import Request

from src.internal.mcp_server.auth import AgenticSearchTokenVerifier
from src.internal.servers.users.api import resolve_request_user


def _response(payload: object, status_code: int = 200) -> Mock:
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _request_with_bearer(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/search/send-search-message",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


@pytest.mark.asyncio
async def test_exchanges_opaque_me_credential_for_resolvable_downstream_jwt() -> None:
    original_pat = "agentic_search_pat_super-secret"
    client = Mock(
        get=AsyncMock(
            return_value=_response(
                {
                    "id": "pat-user",
                    "email": "pat@example.com",
                    "role": "basic",
                    "is_active": True,
                }
            )
        )
    )

    with (
        patch("src.internal.mcp_server.auth.get_http_client", return_value=client),
        patch(
            "src.internal.mcp_server.auth.build_web_base_url",
            return_value="http://web",
        ),
    ):
        access_token = await AgenticSearchTokenVerifier().verify_token(original_pat)

    assert access_token is not None
    assert access_token.token != original_pat
    assert original_pat not in access_token.token
    user = resolve_request_user(_request_with_bearer(access_token.token))
    assert user is not None
    assert user.id == "pat-user"
    assert user.email == "pat@example.com"
    assert user.metadata["role"] == "basic"
    client.get.assert_awaited_once_with(
        "http://web/me",
        headers={"Authorization": f"Bearer {original_pat}"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"id": ""}, {"id": 42}, {"id": "u", "email": 42}],
)
async def test_malformed_me_identity_fails_closed(payload: object) -> None:
    client = Mock(get=AsyncMock(return_value=_response(payload)))
    with patch("src.internal.mcp_server.auth.get_http_client", return_value=client):
        access_token = await AgenticSearchTokenVerifier().verify_token("opaque-pat")

    assert access_token is None


@pytest.mark.asyncio
async def test_invalid_me_json_fails_closed() -> None:
    response = _response(None)
    response.json.side_effect = ValueError("invalid json")
    client = Mock(get=AsyncMock(return_value=response))
    with patch("src.internal.mcp_server.auth.get_http_client", return_value=client):
        access_token = await AgenticSearchTokenVerifier().verify_token("opaque-pat")

    assert access_token is None
