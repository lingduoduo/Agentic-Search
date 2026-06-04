# src/cli/query.py
"""Enterprise knowledge CLI.

Usage:
    # pre-baked token
    python3 -m src.cli.query "summarise last quarter's results" \
        --token <jwt> --url http://localhost:7860

    # mint token from credentials
    python3 -m src.cli.query "what is our refund policy?" \
        --user-id alice --email alice@corp.com --secret "$AUTH_SECRET"

    # interactive prompt
    python3 -m src.cli.query --token <jwt>
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.cli._auth import resolve_token
from src.cli._client import query_agent
from src.cli._render import console, render_answer_progressive, render_sources


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.cli.query",
        description="Query enterprise knowledge from the command line.",
    )
    p.add_argument(
        "query", nargs="?", help="Search query (prompted interactively if omitted)"
    )
    p.add_argument(
        "--url",
        default="http://localhost:7860",
        metavar="URL",
        help="Web backend base URL (default: http://localhost:7860)",
    )
    p.add_argument(
        "--token", metavar="JWT", help="Pre-generated personal access token / JWT"
    )
    p.add_argument(
        "--user-id",
        dest="user_id",
        metavar="ID",
        help="Personal user ID — used to mint a JWT when --token is absent",
    )
    p.add_argument("--email", metavar="EMAIL", help="Email embedded in the minted JWT")
    p.add_argument(
        "--secret",
        metavar="SECRET",
        help="JWT signing secret (falls back to AUTH_SECRET env var)",
    )
    p.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=5,
        metavar="N",
        help="Number of documents to retrieve (default: 5)",
    )
    p.add_argument(
        "--session-id",
        dest="session_id",
        metavar="ID",
        help="Resume a prior chat session",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    query = args.query or console.input("[bold]Query:[/bold] ").strip()
    if not query:
        console.print("[red]No query provided.[/red]")
        return 1

    try:
        token = resolve_token(args.token, args.user_id, args.email, args.secret)
    except ValueError as exc:
        console.print(f"[red]Auth error:[/red] {exc}")
        return 1

    with console.status("[bold green]Searching enterprise knowledge…", spinner="dots"):
        try:
            result = asyncio.run(
                query_agent(
                    args.url,
                    query,
                    token,
                    top_k=args.top_k,
                    session_id=args.session_id,
                )
            )
        except Exception as exc:
            console.print(f"[red]Request failed:[/red] {exc}")
            return 1

    render_sources(result.documents)
    console.rule("[bold]Answer")
    render_answer_progressive(result.answer)
    console.print(f"[dim]session_id: {result.session_id}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
