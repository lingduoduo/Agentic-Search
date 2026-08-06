"""Unit tests for the Wikipedia / ArXiv / Wayback tools. No live network."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.agents.tool.tool_calling import ToolAgentLoopConfig
from src.internal.tools.public_data import knowledge
from src.internal.tools.public_data._http import PublicDataError


def _run(tool, **arguments):
    response, _raw, _meta = asyncio.run(tool.execute("default", arguments))
    return json.loads(response)


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Dense Passage
      Retrieval</title>
    <summary>We study dense retrieval.</summary>
  </entry>
</feed>
"""

CDX_ROWS = [
    ["timestamp", "original", "statuscode", "mimetype"],
    ["20200101120000", "http://example.com/", "200", "text/html"],
]


def test_wikipedia_returns_citeable_shape(monkeypatch):
    calls = []

    async def _fake_get_json(url, *, params=None, **kwargs):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"pageid": 42, "title": "FAISS"}]}}
        return {"query": {"pages": {"42": {"title": "FAISS", "extract": "A library."}}}}

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    result = _run(knowledge.build_wikipedia_tool(), query="faiss")

    assert result == [
        {
            "title": "FAISS",
            "content": "A library.",
            "url": "https://en.wikipedia.org/?curid=42",
        }
    ]
    assert calls[0]["srsearch"] == "faiss"


def test_wikipedia_no_hits_returns_empty_list(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        return {"query": {"search": []}}

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    assert _run(knowledge.build_wikipedia_tool(), query="zzzz") == []


def test_wikipedia_rejects_bogus_language(monkeypatch):
    async def _unreachable(*args, **kwargs):
        raise AssertionError("must not call out with an unvalidated language")

    monkeypatch.setattr(knowledge, "get_json", _unreachable)

    result = _run(knowledge.build_wikipedia_tool(), query="x", language="en.evil.com")

    assert "error" in result


def test_wikipedia_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("wikipedia is down")

    monkeypatch.setattr(knowledge, "get_json", _fail)

    assert _run(knowledge.build_wikipedia_tool(), query="x") == {
        "error": "wikipedia is down"
    }


def test_arxiv_parses_atom_into_citeable_shape(monkeypatch):
    async def _fake_get_text(url, *, params=None, **kwargs):
        assert params["search_query"] == "all:retrieval"
        return ARXIV_FEED

    monkeypatch.setattr(knowledge, "get_text", _fake_get_text)

    result = _run(knowledge.build_arxiv_tool(), query="retrieval")

    assert result == [
        {
            "title": "Dense Passage Retrieval",
            "content": "We study dense retrieval.",
            "url": "http://arxiv.org/abs/1234.5678v1",
        }
    ]


def test_arxiv_malformed_feed_returns_error(monkeypatch):
    async def _fake_get_text(*args, **kwargs):
        return "not xml at all <"

    monkeypatch.setattr(knowledge, "get_text", _fake_get_text)

    assert "error" in _run(knowledge.build_arxiv_tool(), query="x")


def test_arxiv_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("arxiv is down")

    monkeypatch.setattr(knowledge, "get_text", _fail)

    assert _run(knowledge.build_arxiv_tool(), query="x") == {"error": "arxiv is down"}


def test_wayback_returns_snapshot_rows(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        assert params["url"] == "example.com"
        return CDX_ROWS

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    result = _run(knowledge.build_wayback_tool(), url="example.com")

    assert len(result) == 1
    assert result[0]["url"] == (
        "https://web.archive.org/web/20200101120000/http://example.com/"
    )
    assert "2020-01-01T12:00:00" in result[0]["title"]
    assert "200" in result[0]["content"]


def test_wayback_year_filter_is_sent(monkeypatch):
    captured = {}

    async def _fake_get_json(url, *, params=None, **kwargs):
        captured.update(params)
        return CDX_ROWS

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)
    _run(knowledge.build_wayback_tool(), url="example.com", year=2020)

    assert captured["from"] == "20200101"
    assert captured["to"] == "20201231"


def test_wayback_header_only_response_is_empty(monkeypatch):
    async def _fake_get_json(*args, **kwargs):
        return [["timestamp", "original", "statuscode", "mimetype"]]

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    assert _run(knowledge.build_wayback_tool(), url="nope.test") == []


def test_wayback_upstream_failure_returns_error(monkeypatch):
    async def _fail(*args, **kwargs):
        raise PublicDataError("archive is down")

    monkeypatch.setattr(knowledge, "get_json", _fail)

    assert _run(knowledge.build_wayback_tool(), url="x") == {"error": "archive is down"}


@pytest.mark.parametrize(
    "factory",
    [
        knowledge.build_wikipedia_tool,
        knowledge.build_arxiv_tool,
        knowledge.build_wayback_tool,
    ],
)
def test_knowledge_tools_are_citeable_and_read_only(factory):
    from src.internal.tools import ToolEffect

    tool = factory()
    assert tool.citeable is True
    assert tool.effect is ToolEffect.READ_ONLY
    assert tool.schema.description


def test_wikipedia_skips_entries_missing_from_extracts(monkeypatch):
    # A page id present in the search hits but absent from the extracts
    # response must not surface as a blank {"title": "", ...} source card.
    async def _fake_get_json(url, *, params=None, **kwargs):
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [
                        {"pageid": 1, "title": "Has extract"},
                        {"pageid": 2, "title": "Missing from extracts"},
                    ]
                }
            }
        return {"query": {"pages": {"1": {"title": "Has extract", "extract": "Text."}}}}

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    result = _run(knowledge.build_wikipedia_tool(), query="x")

    assert result == [
        {
            "title": "Has extract",
            "content": "Text.",
            "url": "https://en.wikipedia.org/?curid=1",
        }
    ]


# Regression test for the tool-message truncation coupling: ToolAgentLoop caps
# a whole tool response at ``max_tool_response_length`` and truncates by
# keeping the tail (see MAX_CONTENT_CHARS's docstring in ``_http.py``), so a
# default-argument call to a citeable tool must fit under that cap or the
# model silently loses its highest-ranked results.
_TOOL_RESPONSE_CAP = ToolAgentLoopConfig.max_tool_response_length

_LONG_ABSTRACT = "word " * 300  # ~1500 chars, well over MAX_CONTENT_CHARS


def test_arxiv_default_call_fits_tool_response_cap(monkeypatch):
    entries = "\n".join(
        f"""
  <entry>
    <id>http://arxiv.org/abs/{i}</id>
    <title>Paper {i}</title>
    <summary>{_LONG_ABSTRACT}</summary>
  </entry>"""
        for i in range(3)
    )
    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}
</feed>
"""

    async def _fake_get_text(url, *, params=None, **kwargs):
        return feed

    monkeypatch.setattr(knowledge, "get_text", _fake_get_text)

    response, _raw, _meta = asyncio.run(
        knowledge.build_arxiv_tool().execute("default", {"query": "retrieval"})
    )

    assert len(response) <= _TOOL_RESPONSE_CAP


def test_wikipedia_default_call_fits_tool_response_cap(monkeypatch):
    async def _fake_get_json(url, *, params=None, **kwargs):
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [{"pageid": i, "title": f"Page {i}"} for i in range(3)]
                }
            }
        return {
            "query": {
                "pages": {
                    str(i): {"title": f"Page {i}", "extract": _LONG_ABSTRACT}
                    for i in range(3)
                }
            }
        }

    monkeypatch.setattr(knowledge, "get_json", _fake_get_json)

    response, _raw, _meta = asyncio.run(
        knowledge.build_wikipedia_tool().execute("default", {"query": "x"})
    )

    assert len(response) <= _TOOL_RESPONSE_CAP
