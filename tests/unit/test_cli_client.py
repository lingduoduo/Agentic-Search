# tests/unit/test_cli_client.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli._client import AgentResult, query_agent

FAKE_RESPONSE = {
    "session_id": "sess-123",
    "answer": "The quarterly report shows 12% revenue growth.",
    "citations": ["[1]"],
    "documents": [
        {
            "id": "doc1",
            "citation": "[1]",
            "title": "Q3 Financial Report",
            "content": "Revenue grew 12% year-over-year.",
            "url": "https://internal.corp/reports/q3",
            "score": 0.95,
            "metadata": {},
        }
    ],
    "messages": [],
    "hook_metadata": {},
}


def _make_mock_response(data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_query_agent_returns_agent_result():
    mock_resp = _make_mock_response(FAKE_RESPONSE)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        result = await query_agent(
            "http://localhost:7860", "show me the Q3 report", "my.token", top_k=5
        )

    assert isinstance(result, AgentResult)
    assert result.session_id == "sess-123"
    assert result.answer == "The quarterly report shows 12% revenue growth."
    assert len(result.documents) == 1
    assert result.documents[0]["title"] == "Q3 Financial Report"


@pytest.mark.asyncio
async def test_query_agent_sends_correct_headers_and_body():
    mock_resp = _make_mock_response(FAKE_RESPONSE)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        await query_agent(
            "http://localhost:7860", "q", "tok123", top_k=3, session_id="s1"
        )

    call_kwargs = mock_client.post.call_args
    assert call_kwargs.kwargs["headers"] == {"Authorization": "Bearer tok123"}
    body = call_kwargs.kwargs["json"]
    assert body["query"] == "q"
    assert body["top_k"] == 3
    assert body["session_id"] == "s1"


@pytest.mark.asyncio
async def test_query_agent_raises_on_http_error():
    mock_resp = _make_mock_response({}, status_code=401)
    mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.cli._client.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(Exception, match="401"):
            await query_agent("http://localhost:7860", "q", "bad.token")
