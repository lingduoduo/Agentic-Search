"""FastAPI server for SerpAPI-backed retrieval."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests
import uvicorn

from .search_app import create_search_app, format_document

DEFAULT_SERP_URL = "https://serpapi.com/search"
DEFAULT_SERP_ENGINE = "google"
DEFAULT_TOPK = 3
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class SerpSearchConfig:
    search_url: str = DEFAULT_SERP_URL
    topk: int = DEFAULT_TOPK
    serp_api_key: str | None = None
    serp_engine: str = DEFAULT_SERP_ENGINE
    request_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    batch_workers: int = 4

    def validate(self) -> None:
        if not self.search_url:
            raise ValueError("search_url is required.")
        if not self.serp_api_key:
            raise ValueError("SERP API key is required.")
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")


class SerpSearchEngine:
    def __init__(self, config: SerpSearchConfig):
        config.validate()
        self.config = config

    def _search_query(self, query: str) -> dict[str, Any]:
        response = requests.get(
            self.config.search_url,
            params={
                "engine": self.config.serp_engine,
                "q": query,
                "api_key": self.config.serp_api_key,
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def batch_search(self, queries: list[str]) -> list[list[dict[str, dict[str, str]]]]:
        max_workers = min(max(len(queries), 1), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._search_and_process, queries))

    def _search_and_process(self, query: str) -> list[dict[str, dict[str, str]]]:
        if not query.strip():
            return []
        return self._process_result(self._search_query(query))

    def _process_result(self, search_result: dict[str, Any]) -> list[dict[str, dict[str, str]]]:
        documents: list[dict[str, dict[str, str]]] = []

        answer_box = search_result.get("answer_box", {})
        snippet = answer_box.get("snippet") or answer_box.get("answer")
        if answer_box and snippet:
            documents.append(format_document(answer_box.get("title"), snippet, url=answer_box.get("link")))

        for result in search_result.get("organic_results", [])[: self.config.topk]:
            documents.append(format_document(result.get("title"), result.get("snippet"), url=result.get("link")))

        remaining = max(self.config.topk - len(documents), 0)
        for result in search_result.get("related_questions", [])[:remaining]:
            documents.append(format_document(result.get("question"), result.get("snippet"), url=result.get("link")))

        return documents[: self.config.topk]


def create_app(config: SerpSearchConfig):
    return create_search_app("SerpAPI Search Proxy Server", SerpSearchEngine(config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SerpAPI search server.")
    parser.add_argument("--search_url", type=str, default=os.getenv("SERP_SEARCH_URL", DEFAULT_SERP_URL))
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--serp_api_key", type=str, default=os.getenv("SERP_API_KEY"))
    parser.add_argument("--serp_engine", type=str, default=os.getenv("SERP_ENGINE", DEFAULT_SERP_ENGINE))
    parser.add_argument("--host", type=str, default=os.getenv("SERP_SEARCH_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERP_SEARCH_PORT", str(DEFAULT_PORT))))
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    args = parse_args()
    config = SerpSearchConfig(
        search_url=args.search_url,
        topk=args.topk,
        serp_api_key=args.serp_api_key,
        serp_engine=args.serp_engine,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
