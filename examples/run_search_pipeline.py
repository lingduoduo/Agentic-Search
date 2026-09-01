"""Runnable search pipeline: retrieve, enforce access, assemble answer context.

No server, no model, no database — the corpus is a literal in this file. Every
step calls the code the web backend calls, so what this prints is what the real
pipeline does:

  1. Build the caller's filters   build_user_only_filters -> SearchFilters
  2. Retrieve                     TfidfRetriever.from_docs (the demo backend)
  3. Enforce access               SearchFilters.matches
  4. Assemble answer context      documents_from_search_results
                                  + merge_adjacent_documents

Step 3 is the point. Filters are *sent* to a retrieval backend, and the backends
in this repository honour them — but a backend is not obliged to, so passing
filters down is not enforcement on its own. Every filtered retrieval call in the
web backend is paired with its own check for that reason.

Run it both ways to see what the pairing is worth::

    python3 -m examples.run_search_pipeline
    python3 -m examples.run_search_pipeline --skip-enforcement

A document that declares no ACL is public; only a document that declares one is
restricted. That default is the real one — ``SearchFilters.matches`` implements
it — and it is the opposite of an allowlist.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence

from src.context.models import ContextDocument, ContextSection, SearchFilters
from src.context.preprocessing.access_filters import build_user_only_filters
from src.context.search import SearchResult
from src.context.utils import documents_from_search_results, merge_adjacent_documents
from src.internal.servers.retrieval.demo import TfidfRetriever

ADMIN_GROUP = "search-admins"
OWNER_EMAIL = "owner@example.test"


def demo_corpus() -> list[dict]:
    """Five documents: three readable by anyone, two restricted.

    ``metadata["acl"]`` holds the principals allowed to read a document, in the
    prefixed form ``build_access_filter`` produces for the request side. The two
    ``public-guide`` rows are consecutive chunks of one source, so step 4 has
    something to merge.
    """
    return [
        {
            "id": "guide-0",
            "title": "Dense Retrieval Guide",
            "contents": (
                "Rerank deployment guide: a cross-encoder rescoring stage sits "
                "between dense retrieval and generation."
            ),
            "metadata": {
                "document_id": "public-guide",
                "chunk_id": 0,
                "source_type": "file",
            },
        },
        {
            "id": "guide-1",
            "title": "Dense Retrieval Guide",
            "contents": (
                "Deployment keeps the reranker beside the retrieval server so "
                "the extra hop stays inside the cluster."
            ),
            "metadata": {
                "document_id": "public-guide",
                "chunk_id": 1,
                "source_type": "file",
            },
        },
        {
            "id": "faq-0",
            "title": "Retrieval FAQ",
            "contents": "Common questions about dense and sparse retrieval backends.",
            "metadata": {
                "document_id": "public-faq",
                "chunk_id": 0,
                "source_type": "file",
                "acl": ["public"],
            },
        },
        {
            "id": "runbook-0",
            "title": "Restricted Rerank Runbook",
            "contents": (
                "Rerank deployment in production requires admin approval and a "
                "staged rollout across the retrieval fleet."
            ),
            "metadata": {
                "document_id": "restricted-runbook",
                "chunk_id": 0,
                "source_type": "runbook",
                "acl": [f"group:{ADMIN_GROUP}"],
            },
        },
        {
            "id": "memo-0",
            "title": "Quarterly Retrieval Budget",
            "contents": (
                "The quarterly retrieval budget covers embedding refreshes and "
                "scheduled index rebuilds."
            ),
            "metadata": {
                "document_id": "owner-memo",
                "chunk_id": 0,
                "source_type": "file",
                "acl": [f"email:{OWNER_EMAIL}"],
            },
        },
    ]


def retrieve(query: str, *, topk: int = 5) -> list[SearchResult]:
    """Step 2: rank the corpus with the same retriever the demo server serves."""
    retriever = TfidfRetriever.from_docs(demo_corpus())
    rows = retriever.retrieve([query], topk)
    return [SearchResult.from_api_item(item) for item in rows[0]]


def enforce_access(
    results: Sequence[SearchResult], filters: SearchFilters
) -> list[SearchResult]:
    """Step 3: drop what the caller may not read.

    The retrieval leg above never saw ``filters``; a real backend would receive
    them and is expected to honour them, but this check is what makes the
    guarantee the caller's own.
    """
    return [result for result in results if filters.matches(result.metadata)]


def assemble_sections(documents: Sequence[ContextDocument]) -> list[ContextSection]:
    """Step 4: merge consecutive chunks of one source into answer-ready sections."""
    return merge_adjacent_documents(list(documents))


def run_demo(
    *,
    query: str = "rerank deployment",
    user_id: str | None = None,
    email: str | None = None,
    group_ids: Iterable[str] | None = None,
    topk: int = 5,
    enforce: bool = True,
) -> list[ContextSection]:
    """Run all four steps and return the sections that reach answer context."""
    filters = build_user_only_filters(user_id, email=email, group_ids=group_ids)
    results = retrieve(query, topk=topk)
    if enforce:
        results = enforce_access(results, filters)
    return assemble_sections(documents_from_search_results(results))


def _source_ids(sections: Sequence[ContextSection]) -> list[str]:
    return [str(section.center.metadata.get("document_id")) for section in sections]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", default="rerank deployment")
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--email", default=None, help=f"e.g. {OWNER_EMAIL}")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        dest="group_ids",
        help=f"Repeatable group id, e.g. --group {ADMIN_GROUP}",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--skip-enforcement",
        action="store_true",
        help="Drop step 3 to show what an unchecked backend would hand back",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    filters = build_user_only_filters(
        args.user_id, email=args.email, group_ids=args.group_ids
    )

    print(f"Query   : {args.query}")
    print(f"Caller  : acl={filters.access_acl}")

    retrieved = retrieve(args.query, topk=args.topk)
    print(f"Retrieved ({len(retrieved)}): {[r.metadata.get('id') for r in retrieved]}")

    if args.skip_enforcement:
        print("Enforced: SKIPPED (--skip-enforcement)")
        allowed = retrieved
    else:
        allowed = enforce_access(retrieved, filters)
        dropped = len(retrieved) - len(allowed)
        print(f"Enforced: {len(allowed)} readable, {dropped} dropped")

    sections = assemble_sections(documents_from_search_results(allowed))
    print(f"Sections: {_source_ids(sections)}")
    for section in sections:
        chunks = len(section.documents)
        suffix = f" ({chunks} chunks merged)" if chunks > 1 else ""
        print(f"  - {section.center.title}{suffix}")


if __name__ == "__main__":
    main()
