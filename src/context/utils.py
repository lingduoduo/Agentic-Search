"""Utility functions for search-context assembly."""

from __future__ import annotations

import re

from src.retrieval.context import SearchResult

from .models import ContextDocument
from .models import ContextSection
from .models import SearchContextBundle

_CITATION_RE = re.compile(r"\[(D\d+)\]")


def documents_from_search_results(
    results: list[SearchResult],
    *,
    max_documents: int | None = None,
) -> list[ContextDocument]:
    limited = results[:max_documents] if max_documents else results
    return [
        ContextDocument.from_search_result(result, index=index)
        for index, result in enumerate(limited, 1)
    ]


def merge_adjacent_documents(
    documents: list[ContextDocument],
) -> list[ContextSection]:
    """Merge documents with adjacent numeric chunk metadata when available."""

    if not documents:
        return []

    sections: list[ContextSection] = []
    current: list[ContextDocument] = []
    previous_key: tuple[str, int] | None = None

    for document in documents:
        raw_doc_id = document.metadata.get("document_id", document.id)
        raw_chunk_id = document.metadata.get("chunk_id")
        key = (
            str(raw_doc_id),
            int(raw_chunk_id) if isinstance(raw_chunk_id, int) else -10_000,
        )
        if current and previous_key and key == (previous_key[0], previous_key[1] + 1):
            current.append(document)
        else:
            if current:
                sections.append(_section_from_documents(current))
            current = [document]
        previous_key = key

    if current:
        sections.append(_section_from_documents(current))
    return sections


def build_context_bundle(
    query: str,
    results: list[SearchResult],
    *,
    max_documents: int | None = None,
    merge_adjacent: bool = False,
) -> SearchContextBundle:
    documents = documents_from_search_results(results, max_documents=max_documents)
    sections = merge_adjacent_documents(documents) if merge_adjacent else []
    return SearchContextBundle(query=query, documents=documents, sections=sections)


def extract_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(_CITATION_RE.findall(answer)))


def _section_from_documents(documents: list[ContextDocument]) -> ContextSection:
    return ContextSection(
        center=documents[0],
        documents=documents,
        combined_content="\n".join(document.content for document in documents),
    )
