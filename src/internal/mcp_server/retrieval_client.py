"""Authenticated retrieval client shared by MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .utils import build_web_base_url
from .utils import get_http_client
from .utils import require_access_token


class AuthenticatedRetrievalError(RuntimeError):
    """A retrieval failure whose message is safe to expose through MCP."""


@dataclass(frozen=True)
class AuthenticatedDocument:
    """Normalized document returned by authenticated retrieval."""

    title: str
    url: str | None
    content: str
    score: float
    metadata: dict[str, Any]


def _parse_documents(payload: object) -> list[AuthenticatedDocument]:
    if not isinstance(payload, dict):
        raise AuthenticatedRetrievalError("Search service returned an invalid response")

    raw_documents = payload.get("search_docs")
    if not isinstance(raw_documents, list):
        raise AuthenticatedRetrievalError("Search service returned an invalid response")

    documents: list[AuthenticatedDocument] = []
    try:
        for raw_document in raw_documents:
            if not isinstance(raw_document, dict):
                raise TypeError
            title = raw_document["title"]
            url = raw_document["url"]
            content = raw_document["content"]
            score = raw_document["score"]
            metadata = raw_document["metadata"]
            if (
                not isinstance(title, str)
                or (url is not None and not isinstance(url, str))
                or not isinstance(content, str)
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not isinstance(metadata, dict)
            ):
                raise TypeError
            documents.append(
                AuthenticatedDocument(
                    title=title,
                    url=url,
                    content=content,
                    score=float(score),
                    metadata=metadata,
                )
            )
    except (KeyError, TypeError) as exc:
        raise AuthenticatedRetrievalError(
            "Search service returned an invalid response"
        ) from exc
    return documents


async def authenticated_retrieve(
    query: str,
    *,
    top_k: int,
    document_set_names: list[str] | None = None,
) -> list[AuthenticatedDocument]:
    """Retrieve documents through the authenticated web search endpoint."""
    try:
        token = require_access_token()
    except ValueError as exc:
        raise AuthenticatedRetrievalError("Authentication failed") from exc

    filters = {"document_sets": document_set_names} if document_set_names else None
    try:
        response = await get_http_client().post(
            f"{build_web_base_url()}/search/send-search-message",
            headers={"Authorization": f"Bearer {token.token}"},
            json={
                "search_query": query,
                "filters": filters,
                "run_query_expansion": False,
                "num_hits": top_k,
                "stream": False,
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code == 401:
            message = "Authentication failed"
        elif status_code == 403:
            message = "Access to search results was denied"
        elif status_code >= 500:
            message = "Search service is unavailable"
        else:
            message = "Search request failed"
        raise AuthenticatedRetrievalError(message) from exc
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise AuthenticatedRetrievalError("Could not reach the search service") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthenticatedRetrievalError(
            "Search service returned an invalid response"
        ) from exc
    return _parse_documents(payload)
