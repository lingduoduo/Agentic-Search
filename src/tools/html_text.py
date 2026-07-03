"""HTML-to-text extraction for fetched web pages.

Prefers BeautifulSoup when available and falls back to a stdlib
``HTMLParser``-based parser. Both prefer semantic content containers
(``article`` / ``main`` / ``[role=main]`` / ``.content`` / ``#content``) before
falling back to all ``<p>`` tags. No search-specific state — a generic utility
used by the search tool's page fetcher.
"""

from __future__ import annotations

from html.parser import HTMLParser


def _html_to_text(html: str) -> str:
    try:
        import bs4
    except ImportError:
        return _html_to_text_stdlib(html)

    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer semantic content containers before falling back to all <p> tags
    for selector in ("article", "main", '[role="main"]', ".content", "#content"):
        container = soup.select_one(selector)
        if container:
            paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
            text = "\n".join(p for p in paragraphs if p)
            if text:
                return text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
    return text or soup.get_text(" ", strip=True)


class _SemanticTextParser(HTMLParser):
    """Collect paragraph text, preferring semantic content containers."""

    _SEMANTIC_TAGS = {"article", "main"}
    _IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._semantic_depth = 0
        self._semantic_tags: list[str] = []
        self._ignored_depth = 0
        self._paragraph_depth = 0
        self._current: list[str] = []
        self.semantic_paragraphs: list[str] = []
        self.all_paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        if (
            tag in self._SEMANTIC_TAGS
            or attrs_map.get("role") == "main"
            or "content" in (attrs_map.get("class") or "").split()
            or attrs_map.get("id") == "content"
        ):
            self._semantic_depth += 1
            self._semantic_tags.append(tag)
        if tag == "p" and not self._ignored_depth:
            self._paragraph_depth += 1
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._paragraph_depth:
            text = " ".join(" ".join(self._current).split())
            if text:
                self.all_paragraphs.append(text)
                if self._semantic_depth:
                    self.semantic_paragraphs.append(text)
            self._paragraph_depth -= 1
            self._current = []
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        if self._semantic_tags and tag == self._semantic_tags[-1]:
            self._semantic_tags.pop()
            self._semantic_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._paragraph_depth and not self._ignored_depth:
            self._current.append(data)


def _html_to_text_stdlib(html: str) -> str:
    parser = _SemanticTextParser()
    parser.feed(html)
    paragraphs = parser.semantic_paragraphs or parser.all_paragraphs
    return "\n".join(paragraphs) if paragraphs else " ".join(html.split())
