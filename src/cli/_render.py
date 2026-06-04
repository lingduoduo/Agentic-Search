# src/cli/_render.py
from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def render_sources(documents: list[dict]) -> None:
    """Print a source table. No-op if documents is empty."""
    if not documents:
        return
    table = Table(title="Sources", show_header=True, header_style="bold cyan", box=None)
    table.add_column("Cite", style="dim", width=5)
    table.add_column("Title")
    table.add_column("URL", style="blue")
    for doc in documents:
        table.add_row(
            doc.get("citation") or "",
            doc.get("title") or "—",
            doc.get("url") or "—",
        )
    console.print(table)
    console.print()


def render_answer_progressive(
    answer: str,
    *,
    words_per_second: float = 30.0,
) -> None:
    """Reveal *answer* word-by-word as animated rich Markdown.

    ``words_per_second`` controls animation speed (default: 30 ≈ fast read pace).
    """
    words = answer.split()
    if not words:
        return

    delay = 1.0 / max(words_per_second, 1.0)
    accumulated = ""
    with Live(Markdown(""), console=console, refresh_per_second=20) as live:
        for word in words:
            accumulated += ("" if not accumulated else " ") + word
            live.update(Markdown(accumulated))
            time.sleep(delay)
    console.print()
