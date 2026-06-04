from __future__ import annotations
import json
import subprocess
from unittest.mock import MagicMock, patch

from src.backend.servers.retrieval.browser import (
    BrowserSearchEngine,
    BrowserSearchConfig,
)

FAKE_RESULTS = json.dumps(
    [
        {
            "title": "FAISS - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/FAISS",
            "snippet": "A library for similarity search.",
        },
        {
            "title": "GitHub facebookresearch/faiss",
            "url": "https://github.com/facebookresearch/faiss",
            "snippet": "Efficient similarity search.",
        },
    ]
)


def _make_proc(stdout: str = "", returncode: int = 0):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


def test_search_query_returns_formatted_documents():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=2))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),  # open
            _make_proc(),  # snapshot (page load)
            _make_proc(),  # fill --submit
            _make_proc(),  # snapshot (SERP results)
            _make_proc(FAKE_RESULTS),  # --raw eval
            _make_proc(),  # close
        ]
        results = engine._search_and_process("what is FAISS")

    assert len(results) == 2
    assert results[0]["document"]["title"] == "FAISS - Wikipedia"
    assert results[0]["document"]["url"] == "https://en.wikipedia.org/wiki/FAISS"
    assert "FAISS - Wikipedia" in results[0]["document"]["contents"]


def test_empty_results_when_eval_returns_empty_list():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),  # open
            _make_proc(),  # snapshot
            _make_proc(),  # fill
            _make_proc(),  # snapshot
            _make_proc("[]"),  # eval → empty list
            _make_proc(),  # close
        ]
        results = engine._search_and_process("obscure query xyz")
    assert results == []


def test_subprocess_timeout_returns_empty_and_closes():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=5))
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),
            subprocess.TimeoutExpired(cmd="playwright-cli", timeout=30),
            _make_proc(),  # close in finally block
        ]
        results = engine._search_and_process("query")
    assert results == []
    assert mock_run.call_count == 3  # open, snapshot (timeout), close


def test_topk_truncates_results():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=1))
    many = json.dumps(
        [
            {"title": f"Result {i}", "url": f"https://example{i}.com", "snippet": "x"}
            for i in range(5)
        ]
    )
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(many),
            _make_proc(),
        ]
        results = engine._search_and_process("query")
    assert len(results) == 1
    assert results[0]["document"]["title"] == "Result 0"


def test_batch_search_runs_queries_in_parallel():
    engine = BrowserSearchEngine(BrowserSearchConfig(topk=2, batch_workers=2))
    single = json.dumps([{"title": "T", "url": "https://t.com", "snippet": "s"}])
    with patch("src.backend.servers.retrieval.browser.subprocess.run") as mock_run:
        # 6 calls per query × 2 queries = 12 calls
        mock_run.side_effect = [
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(single),
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(),
            _make_proc(single),
            _make_proc(),
        ]
        results = engine.batch_search(["q1", "q2"])
    assert len(results) == 2
    assert results[0][0]["document"]["title"] == "T"
