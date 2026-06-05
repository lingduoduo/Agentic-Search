"""Utility functions for search-context assembly."""

from __future__ import annotations

import re
from urllib.parse import urlparse

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


def _doc_source(doc: ContextDocument) -> str:
    """Stable source identifier for MMR diversity — URL domain or document title."""
    if doc.url:
        return urlparse(doc.url).netloc or doc.url[:50]
    return doc.title or doc.content[:40]


def mmr_rerank(
    docs: list[ContextDocument],
    *,
    topk: int,
    lambda_: float = 0.7,
) -> list[ContextDocument]:
    """Maximal Marginal Relevance reranking on a list of ContextDocuments.

    Selects up to ``topk`` documents greedily: each pick maximises
    ``lambda_ * relevance - (1 - lambda_) * similarity_to_already_selected``.
    Similarity is binary: 1.0 when the candidate shares its source with any
    already-selected doc, 0.0 otherwise.  This penalises back-to-back results
    from the same domain / document title while still preferring high-scoring docs.

    ``lambda_ = 1.0`` degrades to pure score-ordered truncation (no-op).
    ``lambda_ = 0.0`` maximises source diversity regardless of score.
    """
    if not docs or lambda_ >= 1.0:
        return docs[:topk]

    max_score = max(d.score for d in docs) or 1.0
    normalized = [(d, d.score / max_score) for d in docs]

    selected: list[ContextDocument] = []
    selected_sources: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < topk:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:

            def _mmr(
                item: tuple[ContextDocument, float], _sel: list[str] = selected_sources
            ) -> float:
                doc, rel = item
                sim = 1.0 if _doc_source(doc) in _sel else 0.0
                return lambda_ * rel - (1.0 - lambda_) * sim

            best = max(remaining, key=_mmr)

        doc, _ = best
        selected.append(doc)
        selected_sources.append(_doc_source(doc))
        remaining.remove(best)

    return selected


def _section_from_documents(documents: list[ContextDocument]) -> ContextSection:
    return ContextSection(
        center=documents[0],
        documents=documents,
        combined_content="\n".join(document.content for document in documents),
    )
