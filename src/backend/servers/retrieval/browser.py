"""FastAPI retrieval server that uses playwright-cli for browser-based search."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from fastapi import FastAPI

from .app import (
    add_host_port_args,
    create_search_app,
    format_document,
    load_environment,
    run_uvicorn_app,
)

logger = logging.getLogger(__name__)

DEFAULT_TOPK = 5
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PLAYWRIGHT_CMD = "playwright-cli"
SEARCH_URL = "https://www.google.com"
SUBPROCESS_TIMEOUT = 30

# Extracts organic results from a Google SERP via h3 headings — more stable than class names.
_EXTRACT_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('h3')]"
    ".filter(h=>h.closest('a'))"
    ".slice(0,10)"
    ".map(h=>({title:h.textContent.trim(),"
    "url:h.closest('a').href,"
    "snippet:(h.closest('[data-hveid]')?.lastElementChild?.textContent?.trim()||'')}))"
    ".filter(r=>r.url&&!r.url.includes('google.com/search'))"
    ")"
)


@dataclass(frozen=True)
class BrowserSearchConfig:
    topk: int = DEFAULT_TOPK
    batch_workers: int = 4
    subprocess_timeout: int = SUBPROCESS_TIMEOUT


class BrowserSearchEngine:
    def __init__(self, config: BrowserSearchConfig):
        self.config = config

    def _run(
        self,
        *args: str,
        session: str | None = None,
        raw: bool = False,
    ) -> subprocess.CompletedProcess:
        # --raw is a global flag; must precede -s= per playwright-cli CLI contract.
        cmd = [PLAYWRIGHT_CMD]
        if raw:
            cmd.append("--raw")
        if session:
            cmd.append(f"-s={session}")
        cmd.extend(args)
        return subprocess.run(
            cmd,
            capture_output=raw,
            text=True,
            timeout=self.config.subprocess_timeout,
        )

    def _search_and_process(self, query: str) -> list[dict[str, dict[str, str]]]:
        session = f"search-{uuid.uuid4().hex[:8]}"
        try:
            self._run("open", SEARCH_URL, "--persistent", session=session)
            self._run("snapshot", session=session)
            self._run(
                "fill",
                "getByRole('combobox', { name: 'Search' })",
                query,
                "--submit",
                session=session,
            )
            self._run("snapshot", session=session)
            proc = self._run("eval", _EXTRACT_JS, session=session, raw=True)
            raw = json.loads(proc.stdout.strip()) if proc.stdout.strip() else []
            hits = (
                [h for h in raw if isinstance(h, dict)] if isinstance(raw, list) else []
            )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
            logger.warning("browser search failed for %r: %s", query, exc)
            hits = []
        finally:
            try:
                self._run("close", session=session)
            except Exception:
                pass

        return [
            format_document(h.get("title"), h.get("snippet"), h.get("url"))
            for h in hits[: self.config.topk]
        ]

    def batch_search(self, queries: list[str]) -> list[list[dict[str, dict[str, str]]]]:
        max_workers = min(max(len(queries), 1), self.config.batch_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._search_and_process, queries))


def create_app(config: BrowserSearchConfig) -> FastAPI:
    return create_search_app(
        "Browser Retrieval (playwright-cli)", BrowserSearchEngine(config)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser-based retrieval server (playwright-cli)"
    )
    add_host_port_args(
        parser,
        "BROWSER_RETRIEVAL_HOST",
        "BROWSER_RETRIEVAL_PORT",
        DEFAULT_HOST,
        DEFAULT_PORT,
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    config = BrowserSearchConfig(topk=args.topk, batch_workers=args.workers)
    app = create_app(config)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
