"""Web-based connector implementations.

WebConnector   — crawls one or more seed URLs and follows internal links.
RSSConnector   — ingests items from an RSS/Atom feed as documents.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .interface import GenerateDocumentsOutput, LoadConnector, batched
from .models import Document

_SESSION_HEADERS = {
    "User-Agent": "AgenticSearch/1.0 (+https://github.com/lingduoduo/Agentic-Search)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_DEFAULT_TIMEOUT = 10
_DEFAULT_WEB_BATCH = 16
_DEFAULT_RSS_BATCH = 32
_RSS_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def _doc_id(url: str) -> str:
    return "web:" + hashlib.sha1(url.encode()).hexdigest()[:16]


def _same_origin(base: str, url: str) -> bool:
    b, u = urlparse(base), urlparse(url)
    return b.scheme == u.scheme and b.netloc == u.netloc


def _extract_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, body_text) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or url
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    body = soup.get_text(separator="\n", strip=True)
    return title, body


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).split("#")[0].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https"):
            links.append(absolute)
    return links


@dataclass
class WebConnector(LoadConnector):
    """Crawl one or more seed URLs and index the text content.

    Only pages within the same origin as each seed URL are followed.
    Set *max_pages* to limit total crawl size (default 50).
    Set *max_depth* to limit link-following depth (default 2).
    """

    urls: list[str] = field(default_factory=list)
    max_pages: int = 50
    max_depth: int = 2
    timeout: int = _DEFAULT_TIMEOUT
    batch_size: int = _DEFAULT_WEB_BATCH

    def __init__(
        self,
        urls: list[str],
        *,
        max_pages: int = 50,
        max_depth: int = 2,
        timeout: int = _DEFAULT_TIMEOUT,
        batch_size: int = _DEFAULT_WEB_BATCH,
    ) -> None:
        self.urls = list(urls)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.batch_size = batch_size

    def validate_connector_settings(self) -> None:
        if not self.urls:
            raise ValueError("WebConnector requires at least one URL.")
        for url in self.urls:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Unsupported URL scheme: {url!r}")

    def load_from_state(self) -> GenerateDocumentsOutput:
        self.validate_connector_settings()
        session = requests.Session()
        session.headers.update(_SESSION_HEADERS)
        yield from batched(self._crawl(session), self.batch_size)  # type: ignore[misc]

    def _crawl(self, session: requests.Session) -> Iterator[Document]:
        visited: set[str] = set()
        # Queue of (url, depth)
        queue: deque[tuple[str, int]] = deque((url, 0) for url in self.urls)
        while queue and len(visited) < self.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            try:
                resp = session.get(url, timeout=self.timeout, allow_redirects=True)
                resp.raise_for_status()
            except requests.RequestException:
                continue
            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type:
                continue
            title, text = _extract_text(resp.text, url)
            if not text.strip():
                continue
            yield Document(
                id=_doc_id(url),
                title=title,
                url=url,
                contents=text,
                metadata={"source": "web", "crawl_depth": depth},
            )
            if depth < self.max_depth:
                for link in _extract_links(resp.text, url):
                    if link not in visited and _same_origin(url, link):
                        queue.append((link, depth + 1))


class RSSConnector(LoadConnector):
    """Ingest items from an RSS 2.0 or Atom feed as documents."""

    def __init__(
        self,
        feed_url: str,
        *,
        max_items: int = 100,
        batch_size: int = _DEFAULT_RSS_BATCH,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.feed_url = feed_url
        self.max_items = max_items
        self.batch_size = batch_size
        self.timeout = timeout

    def validate_connector_settings(self) -> None:
        if not self.feed_url:
            raise ValueError("RSSConnector requires a feed_url.")

    def load_from_state(self) -> GenerateDocumentsOutput:
        self.validate_connector_settings()
        yield from batched(self._iter_items(), self.batch_size)  # type: ignore[misc]

    def _iter_items(self) -> Iterator[Document]:
        resp = requests.get(
            self.feed_url,
            headers=_SESSION_HEADERS,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse RSS feed: {exc}") from exc

        count = 0
        for item in self._iter_entries(root):
            if count >= self.max_items:
                break
            doc = self._entry_to_document(item, root.tag)
            if doc:
                yield doc
                count += 1

    def _iter_entries(self, root: ET.Element) -> Iterator[ET.Element]:
        tag = root.tag.lower()
        if "feed" in tag:
            # Atom
            ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
            prefix = f"{{{ns}}}" if ns else ""
            yield from root.findall(f"{prefix}entry")
        else:
            # RSS 2.0
            channel = root.find("channel")
            if channel is not None:
                yield from channel.findall("item")

    def _entry_to_document(self, entry: ET.Element, feed_tag: str) -> Document | None:
        def _find(tag: str) -> str:
            el = entry.find(tag)
            return (el.text or "").strip() if el is not None else ""

        is_atom = "feed" in feed_tag.lower()
        if is_atom:
            ns = feed_tag.split("}")[0].lstrip("{") if "}" in feed_tag else ""
            prefix = f"{{{ns}}}" if ns else ""
            title = _find(f"{prefix}title") or "Untitled"
            link_el = entry.find(f"{prefix}link")
            url = link_el.get("href", "") if link_el is not None else ""
            summary_el = entry.find(f"{prefix}summary") or entry.find(
                f"{prefix}content"
            )
            text = (summary_el.text or "").strip() if summary_el is not None else ""
            published = _find(f"{prefix}published") or _find(f"{prefix}updated")
        else:
            title = _find("title") or "Untitled"
            url = _find("link") or _find("guid")
            text = _find("description")
            # content:encoded
            encoded = entry.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            if encoded is not None and encoded.text:
                text = encoded.text.strip()
            published = _find("pubDate")

        if not text:
            return None

        # Strip HTML from summary/description
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text(separator="\n", strip=True)
        doc_id = _doc_id(url or title)
        meta: dict[str, Any] = {"source": "rss", "feed_url": self.feed_url}
        if published:
            meta["published"] = published
        return Document(
            id=doc_id,
            title=title,
            url=url or None,
            contents=clean_text,
            metadata=meta,
        )
