"""FastAPI server for Google Custom Search retrieval."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import aiohttp
import bs4
import uvicorn
from googleapiclient.discovery import build

from .search_app import create_search_app, format_document

try:
    import chardet
except ImportError:  # pragma: no cover - fallback for lean environments
    chardet = None

DEFAULT_TOPK = 3
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_GOOGLE_RESULTS_PER_PAGE = 10

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
]


@dataclass(frozen=True)
class OnlineSearchConfig:
    topk: int = DEFAULT_TOPK
    api_key: str | None = None
    cse_id: str | None = None
    snippet_only: bool = False
    request_timeout_seconds: int = 5
    fetch_concurrency: int = 8
    batch_workers: int = 4

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required.")
        if not self.cse_id:
            raise ValueError("Google CSE ID is required.")
        if self.topk < 1:
            raise ValueError("topk must be at least 1.")

def parse_snippet(snippet: str) -> list[str]:
    return [segment.strip() for segment in snippet.split("...") if len(segment.strip().split()) > 5]


def sanitize_search_query(query: str) -> str:
    sanitized = re.sub(r"[^\w\s]", " ", query)
    sanitized = re.sub(r"[\t\r\f\v\n]", " ", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()


def filter_links(search_results: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []
    for result in search_results:
        for item in result.get("items", []):
            link = item.get("link", "")
            if not link or "mime" in item:
                continue
            extension = os.path.splitext(link)[1].lower()
            if extension in {"", ".html", ".htm", ".shtml"}:
                links.append(link)
    return links


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> str:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    async with semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                raw = await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return ""

    if chardet is None:
        encoding = "utf-8"
    else:
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
    return raw.decode(encoding, errors="ignore")


async def fetch_all(urls: list[str], timeout_seconds: int, limit: int) -> list[str]:
    if not urls:
        return []

    semaphore = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        config.validate()
        self.config = config
        self._service = build("customsearch", "v1", developerKey=self.config.api_key)

    def collect_context(self, snippet: str, document: str) -> str:
        normalized_document = document.replace("\r", "")
        paragraphs = [paragraph.strip() for paragraph in normalized_document.split("\n") if paragraph.strip()]
        lowered_paragraphs = [paragraph.lower() for paragraph in paragraphs]

        contexts: list[str] = []
        for snippet_part in parse_snippet(snippet):
            target = snippet_part.lower()
            for paragraph, lowered in zip(paragraphs, lowered_paragraphs):
                if target in lowered and paragraph not in contexts:
                    contexts.append(paragraph)
                    break

        return "\n".join(contexts)

    def fetch_web_content(self, search_results: list[dict[str, Any]]) -> dict[str, str]:
        links = filter_links(search_results)
        if not links:
            return {}
        html_documents = asyncio.run(
            fetch_all(
                urls=links,
                timeout_seconds=self.config.request_timeout_seconds,
                limit=self.config.fetch_concurrency,
            )
        )

        content_by_link: dict[str, str] = {}
        for html, link in zip(html_documents, links):
            if not html:
                continue
            soup = bs4.BeautifulSoup(html, "html.parser")
            paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
            text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
            if text:
                content_by_link[link] = text

        return content_by_link

    def search(self, search_term: str, num_pages: int = 1) -> list[dict[str, Any]]:
        query = sanitize_search_query(search_term)
        if not query:
            return []

        results: list[dict[str, Any]] = []
        response = self._service.cse().list(
            q=query,
            cx=self.config.cse_id,
            num=min(self.config.topk, MAX_GOOGLE_RESULTS_PER_PAGE),
        ).execute()
        results.append(response)

        for _ in range(max(num_pages - 1, 0)):
            next_pages = response.get("queries", {}).get("nextPage")
            if not next_pages:
                break
            start_index = next_pages[0].get("startIndex")
            if not start_index:
                break
            response = self._service.cse().list(
                q=query,
                cx=self.config.cse_id,
                start=start_index,
                num=min(self.config.topk, MAX_GOOGLE_RESULTS_PER_PAGE),
            ).execute()
            results.append(response)

        return results

    def batch_search(self, queries: list[str]) -> list[list[dict[str, dict[str, str]]]]:
        max_workers = min(max(len(queries), 1), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._retrieve_context, queries))

    def _retrieve_context(self, query: str) -> list[dict[str, dict[str, str]]]:
        search_results = self.search(query)
        if not search_results:
            return []

        contexts: list[dict[str, dict[str, str]]] = []
        content_dict = {} if self.config.snippet_only else self.fetch_web_content(search_results)

        for result in search_results:
            for item in result.get("items", []):
                title = item.get("title") or "No title."
                snippet = item.get("snippet", "")

                if self.config.snippet_only:
                    context = " ".join(parse_snippet(snippet)) or "No snippet available."
                else:
                    link = item.get("link", "")
                    document = content_dict.get(link, "")
                    context = self.collect_context(snippet, document) or "No snippet available."

                if title == "No title." and context == "No snippet available.":
                    continue

                contexts.append(format_document(title, context))
                if len(contexts) >= self.config.topk:
                    return contexts

        return contexts


def create_app(config: OnlineSearchConfig):
    return create_search_app("Google Search Proxy Server", OnlineSearchEngine(config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch online search server.")
    parser.add_argument("--api_key", type=str, default=os.getenv("GOOGLE_API_KEY"), help="API key for Google search")
    parser.add_argument("--cse_id", type=str, default=os.getenv("GOOGLE_CSE_ID"), help="CSE ID for Google search")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of results to return per query")
    parser.add_argument(
        "--snippet_only",
        action="store_true",
        help="If set, only return snippets; otherwise, return fetched page context.",
    )
    parser.add_argument("--host", type=str, default=os.getenv("GOOGLE_SEARCH_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("GOOGLE_SEARCH_PORT", str(DEFAULT_PORT))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OnlineSearchConfig(
        api_key=args.api_key,
        cse_id=args.cse_id,
        topk=args.topk,
        snippet_only=args.snippet_only,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
