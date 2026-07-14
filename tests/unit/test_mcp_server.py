"""Unit tests for the MCP server tools, resources, and utils."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from unittest.mock import Mock

import pytest

from src.tools.search import SearchPage
from src.internal.mcp_server.retrieval_client import AuthenticatedDocument
from src.internal.mcp_server.retrieval_client import AuthenticatedRetrievalError


@pytest.mark.parametrize("module_name", ["search", "research", "chat"])
def test_indexed_document_tools_do_not_import_raw_retrieval(module_name: str) -> None:
    """Indexed-document MCP tools must cross the authenticated client boundary."""
    module_path = (
        Path(__file__).parents[2]
        / "src"
        / "internal"
        / "mcp_server"
        / "tools"
        / f"{module_name}.py"
    )
    tree = ast.parse(module_path.read_text())
    required_symbol = "authenticated_retrieve"
    forbidden_module_prefixes = (
        "src.context.retrieval",
        "src.internal.document_index.retrieval",
        "src.internal.retrieval",
        "src.internal.servers.retrieval",
    )
    forbidden_symbols = {
        "answer_with_retrieval",
        "retrieval_search",
        "retrieve_context",
        "run_search",
        "search_chunks",
    }
    imported_symbols: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_symbols.update(imported)
            assert not (imported & forbidden_symbols)
            assert not (node.module or "").startswith(forbidden_module_prefixes)
        elif isinstance(node, ast.Import):
            imported_modules = {alias.name for alias in node.names}
            assert not any(
                imported_module.startswith(forbidden_module_prefixes)
                for imported_module in imported_modules
            )

    assert required_symbol in imported_symbols


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_build_web_base_url_defaults():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(os.environ, {}, clear=False):
        for var in (
            "API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS",
            "API_SERVER_PROTOCOL",
            "API_SERVER_HOST",
            "AGENTIC_SEARCH_WEB_PORT",
        ):
            os.environ.pop(var, None)
        url = build_web_base_url()
    assert url == "http://127.0.0.1:7860"


def test_build_web_base_url_override():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(
        os.environ,
        {"API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS": "https://my.server/"},
    ):
        url = build_web_base_url()
    assert url == "https://my.server"  # trailing slash stripped


def test_build_web_base_url_custom_port():
    from src.internal.mcp_server.utils import build_web_base_url

    with patch.dict(os.environ, {"AGENTIC_SEARCH_WEB_PORT": "9999"}, clear=False):
        os.environ.pop("API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS", None)
        url = build_web_base_url()
    assert url.endswith(":9999")


# ---------------------------------------------------------------------------
# api helpers
# ---------------------------------------------------------------------------


def test_get_cors_origins_empty():
    from src.internal.mcp_server.api import _get_cors_origins

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MCP_SERVER_CORS_ORIGINS", None)
        assert _get_cors_origins() == []


def test_get_cors_origins_multiple():
    from src.internal.mcp_server.api import _get_cors_origins

    with patch.dict(
        os.environ,
        {"MCP_SERVER_CORS_ORIGINS": "https://a.com, https://b.com"},
    ):
        result = _get_cors_origins()
    assert result == ["https://a.com", "https://b.com"]


# ---------------------------------------------------------------------------
# tools/search — search_indexed_documents
# ---------------------------------------------------------------------------


_FAKE_PAGES = [
    SearchPage(
        title="Dense Retrieval",
        summary="FAISS-based dense retrieval.",
        url="http://ex.com/1",
    ),
    SearchPage(
        title="Sparse BM25",
        summary="BM25 scoring for keyword search.",
        url="http://ex.com/2",
    ),
]

_AUTHENTICATED_DOCUMENTS = [
    AuthenticatedDocument(
        title="Dense Retrieval",
        content="FAISS-based dense retrieval.",
        url="http://ex.com/1",
        score=0.9,
        metadata={},
    ),
    AuthenticatedDocument(
        title="Sparse BM25",
        content="BM25 scoring for keyword search.",
        url="http://ex.com/2",
        score=0.8,
        metadata={},
    ),
]


@pytest.mark.asyncio
async def test_search_indexed_documents_returns_results():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    authenticated_retrieve = AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS)
    with patch(
        "src.internal.mcp_server.tools.search.authenticated_retrieve",
        authenticated_retrieve,
        create=True,
    ):
        result = await search_indexed_documents(
            query="dense retrieval", document_set_names=[]
        )

    authenticated_retrieve.assert_awaited_once_with(
        "dense retrieval", top_k=5, document_set_names=None
    )

    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Dense Retrieval"
    assert result["results"][0]["url"] == "http://ex.com/1"
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_indexed_documents_empty():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with patch(
        "src.internal.mcp_server.tools.search.authenticated_retrieve",
        new=AsyncMock(return_value=[]),
        create=True,
    ):
        result = await search_indexed_documents(query="nothing")

    assert result == {"results": []}


@pytest.mark.asyncio
async def test_search_indexed_documents_error():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with patch(
        "src.internal.mcp_server.tools.search.authenticated_retrieve",
        new=AsyncMock(side_effect=AuthenticatedRetrievalError("Authentication failed")),
        create=True,
    ):
        result = await search_indexed_documents(query="test")

    assert "error" in result
    assert "Authentication failed" in result["error"]


@pytest.mark.asyncio
async def test_search_indexed_documents_authorization_error_has_no_raw_fallback():
    from src.internal.mcp_server.tools.search import search_indexed_documents

    with (
        patch(
            "src.internal.mcp_server.tools.search.authenticated_retrieve",
            new=AsyncMock(
                side_effect=AuthenticatedRetrievalError(
                    "Access to search results was denied"
                )
            ),
            create=True,
        ),
        patch("src.tools.search.retrieval_search", new=AsyncMock()) as raw_retrieve,
    ):
        result = await search_indexed_documents(query="private")

    raw_retrieve.assert_not_awaited()
    assert result == {
        "error": "Document search failed: Access to search results was denied",
        "results": [],
    }


# ---------------------------------------------------------------------------
# tools/research — retrieve_documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_documents_forwards_query_and_top_k():
    from src.internal.mcp_server.tools.research import retrieve_documents

    authenticated_retrieve = AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS[:1])
    with patch(
        "src.internal.mcp_server.tools.research.authenticated_retrieve",
        authenticated_retrieve,
        create=True,
    ):
        result = await retrieve_documents("dense retrieval", top_k=8)

    authenticated_retrieve.assert_awaited_once_with("dense retrieval", top_k=8)
    assert result == {
        "documents": [
            {
                "id": "D1",
                "title": "Dense Retrieval",
                "url": "http://ex.com/1",
                "content": "FAISS-based dense retrieval.",
                "score": 0.9,
            }
        ],
        "query": "dense retrieval",
    }


@pytest.mark.asyncio
async def test_retrieve_documents_empty_is_success():
    from src.internal.mcp_server.tools.research import retrieve_documents

    with patch(
        "src.internal.mcp_server.tools.research.authenticated_retrieve",
        new=AsyncMock(return_value=[]),
        create=True,
    ):
        result = await retrieve_documents("nothing")

    assert result == {"documents": [], "query": "nothing"}


@pytest.mark.asyncio
async def test_retrieve_documents_authentication_error_has_no_raw_fallback():
    from src.internal.mcp_server.tools.research import retrieve_documents

    with (
        patch(
            "src.internal.mcp_server.tools.research.authenticated_retrieve",
            new=AsyncMock(
                side_effect=AuthenticatedRetrievalError("Authentication failed")
            ),
            create=True,
        ),
        patch("src.context.pipeline.retrieve_context", new=AsyncMock()) as raw_retrieve,
    ):
        result = await retrieve_documents("private")

    raw_retrieve.assert_not_awaited()
    assert result == {
        "error": "Authentication failed",
        "documents": [],
        "query": "private",
    }


# ---------------------------------------------------------------------------
# tools/chat — ask_agentic_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_agentic_search_synthesizes_only_from_authorized_documents():
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    authenticated_retrieve = AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS)

    def synthesize(request: object, *, llm: object) -> Mock:
        return Mock(
            answer="Dense and sparse retrieval are supported [D1] [D2].",
            citations=["D1", "D2"],
            context=request.context,
        )

    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            authenticated_retrieve,
            create=True,
        ),
        patch(
            "src.internal.mcp_server.tools.chat.generate_answer",
            side_effect=synthesize,
            create=True,
        ) as generate,
        patch(
            "src.internal.mcp_server.tools.chat.answer_with_retrieval",
            new=AsyncMock(),
            create=True,
        ) as raw_answer,
    ):
        result = await ask_agentic_search("Which retrieval methods?", top_k=2)

    authenticated_retrieve.assert_awaited_once_with("Which retrieval methods?", top_k=2)
    raw_answer.assert_not_awaited()
    request = generate.call_args.args[0]
    assert request.verify_grounding is True
    assert [document.title for document in request.context.documents] == [
        "Dense Retrieval",
        "Sparse BM25",
    ]
    assert [document.content for document in request.context.documents] == [
        "FAISS-based dense retrieval.",
        "BM25 scoring for keyword search.",
    ]
    assert result == {
        "answer": "Dense and sparse retrieval are supported [D1] [D2].",
        "citations": ["D1", "D2"],
        "sources": [
            {
                "title": "Dense Retrieval",
                "url": "http://ex.com/1",
                "content": "FAISS-based dense retrieval.",
            },
            {
                "title": "Sparse BM25",
                "url": "http://ex.com/2",
                "content": "BM25 scoring for keyword search.",
            },
        ],
    }


@pytest.mark.asyncio
async def test_ask_agentic_search_empty_evidence_skips_llm():
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    llm = Mock()
    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(return_value=[]),
            create=True,
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm", return_value=llm),
        patch(
            "src.internal.mcp_server.tools.chat.answer_with_retrieval",
            new=AsyncMock(),
            create=True,
        ) as raw_answer,
    ):
        result = await ask_agentic_search("unknown")

    llm.complete.assert_not_called()
    raw_answer.assert_not_awaited()
    assert result == {
        "answer": "I could not find retrieved context to answer: unknown",
        "citations": [],
        "sources": [],
    }


@pytest.mark.asyncio
async def test_ask_agentic_search_whitespace_evidence_skips_llm_construction():
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    whitespace_document = AuthenticatedDocument(
        title="Empty authorized document",
        content="  \n\t ",
        url="http://ex.com/empty",
        score=0.7,
        metadata={},
    )
    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(return_value=[whitespace_document]),
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm") as build_llm,
    ):
        result = await ask_agentic_search("unknown")

    build_llm.assert_not_called()
    assert result == {
        "answer": "I could not find retrieved context to answer: unknown",
        "citations": [],
        "sources": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message", ["Authentication failed", "Access to search results was denied"]
)
async def test_ask_agentic_search_auth_errors_have_no_raw_fallback(message: str):
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    llm = Mock()
    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(side_effect=AuthenticatedRetrievalError(message)),
            create=True,
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm", return_value=llm),
        patch(
            "src.internal.mcp_server.tools.chat.answer_with_retrieval",
            new=AsyncMock(),
            create=True,
        ) as raw_answer,
    ):
        result = await ask_agentic_search("private")

    llm.complete.assert_not_called()
    raw_answer.assert_not_awaited()
    assert result == {
        "error": message,
        "answer": "",
        "citations": [],
        "sources": [],
    }


@pytest.mark.asyncio
async def test_ask_agentic_search_uses_llm_only_after_authenticated_evidence():
    from src.context import LLMResponse
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    llm = Mock()
    llm.complete.return_value = LLMResponse("FAISS provides dense retrieval [D1].")
    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS[:1]),
            create=True,
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm", return_value=llm),
        patch(
            "src.internal.mcp_server.tools.chat.answer_with_retrieval",
            new=AsyncMock(),
            create=True,
        ) as raw_answer,
    ):
        result = await ask_agentic_search("What provides dense retrieval?")

    raw_answer.assert_not_awaited()
    llm.complete.assert_called_once()
    prompt_text = "\n".join(
        message.content for message in llm.complete.call_args.args[0]
    )
    assert "FAISS-based dense retrieval." in prompt_text
    assert "BM25 scoring" not in prompt_text
    assert result["answer"] == "FAISS provides dense retrieval [D1]."
    assert result["citations"] == ["D1"]


@pytest.mark.asyncio
async def test_ask_agentic_search_verifies_adversarial_llm_citations():
    from src.context import LLMResponse
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    llm = Mock()
    llm.complete.return_value = LLMResponse(
        "FAISS provides dense retrieval [D1]. The moon is cheese [D99]."
    )
    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS[:1]),
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm", return_value=llm),
    ):
        result = await ask_agentic_search("What provides dense retrieval?")

    assert result["answer"] == (
        "FAISS provides dense retrieval [D1]. The moon is cheese ."
    )
    assert result["citations"] == ["D1"]
    assert result["sources"] == [
        {
            "title": "Dense Retrieval",
            "url": "http://ex.com/1",
            "content": "FAISS-based dense retrieval.",
        }
    ]


@pytest.mark.asyncio
async def test_ask_agentic_search_uses_extractive_authenticated_evidence():
    from src.internal.mcp_server.tools.chat import ask_agentic_search

    with (
        patch(
            "src.internal.mcp_server.tools.chat.authenticated_retrieve",
            new=AsyncMock(return_value=_AUTHENTICATED_DOCUMENTS[:1]),
            create=True,
        ),
        patch("src.internal.mcp_server.tools.chat._build_llm", return_value=None),
        patch(
            "src.internal.mcp_server.tools.chat.answer_with_retrieval",
            new=AsyncMock(),
            create=True,
        ) as raw_answer,
    ):
        result = await ask_agentic_search("What provides dense retrieval?")

    raw_answer.assert_not_awaited()
    assert result["answer"] == "FAISS-based dense retrieval. [D1]"
    assert result["citations"] == ["D1"]
    assert result["sources"] == [
        {
            "title": "Dense Retrieval",
            "url": "http://ex.com/1",
            "content": "FAISS-based dense retrieval.",
        }
    ]


# ---------------------------------------------------------------------------
# tools/search — search_web
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_web_google_provider():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(return_value=_FAKE_PAGES),
        ):
            result = await search_web(query="FAISS", limit=2)

    assert result["query"] == "FAISS"
    assert len(result["results"]) == 2
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_web_serpapi_provider():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "serpapi"}):
        with patch(
            "src.internal.mcp_server.tools.search.serpapi_search",
            new=AsyncMock(return_value=_FAKE_PAGES),
        ):
            result = await search_web(query="BM25")

    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_search_web_filters_error_pages():
    from src.internal.mcp_server.tools.search import search_web

    pages_with_error = [
        SearchPage(title="OK", summary="good result", url="http://ex.com/ok"),
        SearchPage(error="rate limited"),
    ]
    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(return_value=pages_with_error),
        ):
            result = await search_web(query="test")

    # Error pages are filtered from results
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "OK"


@pytest.mark.asyncio
async def test_search_web_provider_exception():
    from src.internal.mcp_server.tools.search import search_web

    with patch.dict(os.environ, {"MCP_WEB_SEARCH_PROVIDER": "google"}):
        with patch(
            "src.internal.mcp_server.tools.search.google_custom_search",
            new=AsyncMock(side_effect=RuntimeError("network error")),
        ):
            result = await search_web(query="test")

    assert "error" in result
    assert result["results"] == []


# ---------------------------------------------------------------------------
# tools/search — open_urls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_urls_fetches_each():
    from src.internal.mcp_server.tools.search import open_urls

    async def _fake_fetch(url: str, **kwargs: object) -> str:
        return f"content of {url}"

    with patch(
        "src.internal.mcp_server.tools.search.fetch_url", side_effect=_fake_fetch
    ):
        result = await open_urls(urls=["http://a.com", "http://b.com"])

    assert len(result["results"]) == 2
    assert result["results"][0] == {
        "url": "http://a.com",
        "content": "content of http://a.com",
    }


@pytest.mark.asyncio
async def test_open_urls_error():
    from src.internal.mcp_server.tools.search import open_urls

    with patch(
        "src.internal.mcp_server.tools.search.fetch_url",
        side_effect=RuntimeError("timeout"),
    ):
        result = await open_urls(urls=["http://ex.com"])

    assert "error" in result


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indexed_sources_retrieval_only():
    from src.internal.mcp_server.resources.indexed_sources import (
        indexed_sources_resource,
    )

    env = {
        k: ""
        for k in (
            "GOOGLE_API_KEY",
            "GOOGLE_CSE_ID",
            "SERPAPI_API_KEY",
            "SERP_API_KEY",
            "SERPER_API_KEY",
        )
    }
    with patch.dict(os.environ, env):
        result = json.loads(await indexed_sources_resource())
    assert result == ["retrieval"]


@pytest.mark.asyncio
async def test_indexed_sources_with_google():
    from src.internal.mcp_server.resources.indexed_sources import (
        indexed_sources_resource,
    )

    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "key",
            "GOOGLE_CSE_ID": "cse",
            "SERPER_API_KEY": "",
            "SERPAPI_API_KEY": "",
            "SERP_API_KEY": "",
        },
    ):
        result = json.loads(await indexed_sources_resource())
    assert "google" in result
    assert "retrieval" in result


@pytest.mark.asyncio
async def test_document_sets_returns_empty():
    from src.internal.mcp_server.resources.document_sets import document_sets_resource

    result = json.loads(await document_sets_resource())
    assert result == []
