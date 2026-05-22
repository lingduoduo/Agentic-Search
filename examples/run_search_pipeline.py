"""Runnable search pipeline example with filters and permission filtering.

This file is intentionally lightweight: it mirrors a production search flow
without depending on database models, enterprise ACL packages, or a live vector
index. The core steps are:

1. Build normalized index filters from user filters and persona defaults.
2. Search an index implementation.
3. Apply a permission-filter entry point.
4. Merge adjacent chunks into sections for answer generation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SearchUser:
    """Caller identity used by the permission filter."""

    id: str
    email: str | None = None
    group_ids: frozenset[str] = frozenset()
    is_anonymous: bool = False


@dataclass(frozen=True)
class AccessPolicy:
    """Document-level access policy."""

    public: bool = True
    allowed_user_emails: frozenset[str] = frozenset()
    allowed_group_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SearchDocument:
    """One searchable document in the demo index."""

    id: str
    title: str
    contents: str
    url: str | None = None
    source_type: str = "file"
    document_set: str | None = None
    tags: frozenset[str] = frozenset()
    updated_at: datetime | None = None
    access: AccessPolicy = field(default_factory=AccessPolicy)


@dataclass(frozen=True)
class SearchFilters:
    """User-selected filters."""

    source_type: str | None = None
    document_set: frozenset[str] | None = None
    tags: frozenset[str] = frozenset()
    time_cutoff: datetime | None = None
    attached_document_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PersonaSearchInfo:
    """Persona defaults used when the request does not specify filters."""

    document_set_names: frozenset[str] | None = None
    search_start_date: datetime | None = None
    attached_document_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SearchRequest:
    query: str
    filters: SearchFilters | None = None
    limit: int = 5
    bypass_acl: bool = False
    merge_adjacent_chunks: bool = True


@dataclass(frozen=True)
class IndexFilters:
    """Final filters consumed by the index."""

    source_type: str | None
    document_set: frozenset[str] | None
    tags: frozenset[str]
    time_cutoff: datetime | None
    attached_document_ids: frozenset[str]


@dataclass(frozen=True)
class SearchChunk:
    document_id: str
    chunk_id: int
    text: str
    score: float
    title: str
    url: str | None
    source_type: str
    document_set: str | None
    tags: frozenset[str]
    updated_at: datetime | None
    access: AccessPolicy


@dataclass(frozen=True)
class SearchSection:
    document_id: str
    title: str
    text: str
    score: float
    url: str | None = None


PermissionFilter = Callable[
    [Sequence[SearchChunk], SearchUser, bool], list[SearchChunk]
]


class InMemorySearchIndex:
    """Small lexical index used by the example and unit tests."""

    def __init__(self, documents: Iterable[SearchDocument], *, chunk_size: int = 320):
        self.documents = list(documents)
        self.chunk_size = chunk_size

    def search(
        self,
        query: str,
        *,
        filters: IndexFilters,
        limit: int,
    ) -> list[SearchChunk]:
        query_terms = _tokenize(query)
        chunks: list[SearchChunk] = []
        for document in self.documents:
            if not _matches_filters(document, filters):
                continue
            for chunk_id, text in enumerate(
                _split_text(document.contents, self.chunk_size)
            ):
                score = _score_text(query_terms, f"{document.title} {text}")
                if score <= 0:
                    continue
                chunks.append(
                    SearchChunk(
                        document_id=document.id,
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                        title=document.title,
                        url=document.url,
                        source_type=document.source_type,
                        document_set=document.document_set,
                        tags=document.tags,
                        updated_at=document.updated_at,
                        access=document.access,
                    )
                )
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:limit]


def build_index_filters(
    *,
    user_filters: SearchFilters | None,
    persona_search_info: PersonaSearchInfo | None = None,
) -> IndexFilters:
    """Merge user-selected filters with persona defaults."""

    base = user_filters or SearchFilters()
    persona_document_sets = (
        persona_search_info.document_set_names if persona_search_info else None
    )
    persona_time_cutoff = (
        persona_search_info.search_start_date if persona_search_info else None
    )
    persona_attached_ids = (
        persona_search_info.attached_document_ids
        if persona_search_info
        else frozenset()
    )

    return IndexFilters(
        source_type=base.source_type,
        document_set=base.document_set
        if base.document_set is not None
        else persona_document_sets,
        tags=base.tags,
        time_cutoff=base.time_cutoff or persona_time_cutoff,
        attached_document_ids=base.attached_document_ids or persona_attached_ids,
    )


def default_permission_filter(
    chunks: Sequence[SearchChunk],
    user: SearchUser,
    bypass_acl: bool = False,
) -> list[SearchChunk]:
    """Permission-filter entry point.

    Replace this callable when integrating with a real ACL backend. Anonymous
    users only see public chunks; authenticated users also see chunks granted to
    their email or one of their groups.
    """

    if bypass_acl:
        return list(chunks)

    allowed: list[SearchChunk] = []
    for chunk in chunks:
        access = chunk.access
        if access.public:
            allowed.append(chunk)
            continue
        if user.is_anonymous:
            continue
        if user.email and user.email in access.allowed_user_emails:
            allowed.append(chunk)
            continue
        if user.group_ids & access.allowed_group_ids:
            allowed.append(chunk)
    return allowed


def search_pipeline(
    *,
    request: SearchRequest,
    index: InMemorySearchIndex,
    user: SearchUser,
    persona_search_info: PersonaSearchInfo | None = None,
    permission_filter: PermissionFilter = default_permission_filter,
) -> list[SearchSection]:
    """Execute search, metadata filtering, permission filtering, and merging."""

    if request.limit <= 0:
        return []

    filters = build_index_filters(
        user_filters=request.filters,
        persona_search_info=persona_search_info,
    )
    retrieved_chunks = index.search(
        request.query,
        filters=filters,
        limit=request.limit,
    )
    visible_chunks = permission_filter(retrieved_chunks, user, request.bypass_acl)
    if request.merge_adjacent_chunks:
        return merge_adjacent_chunks(visible_chunks)
    return [section_from_chunks([chunk]) for chunk in visible_chunks]


def merge_adjacent_chunks(chunks: Sequence[SearchChunk]) -> list[SearchSection]:
    """Merge adjacent chunks from the same document while preserving rank order."""

    if not chunks:
        return []

    original_index = {
        (chunk.document_id, chunk.chunk_id): index for index, chunk in enumerate(chunks)
    }
    by_document: dict[str, list[SearchChunk]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)

    chunk_to_section: dict[tuple[str, int], SearchSection] = {}
    chunk_to_section_key: dict[tuple[str, int], tuple[str, int]] = {}
    for document_chunks in by_document.values():
        document_chunks.sort(key=lambda chunk: chunk.chunk_id)
        current_group = [document_chunks[0]]
        for chunk in document_chunks[1:]:
            previous = current_group[-1]
            if chunk.chunk_id == previous.chunk_id + 1:
                current_group.append(chunk)
                continue
            _assign_section(
                current_group,
                original_index,
                chunk_to_section,
                chunk_to_section_key,
            )
            current_group = [chunk]
        _assign_section(
            current_group,
            original_index,
            chunk_to_section,
            chunk_to_section_key,
        )

    seen: set[tuple[str, int]] = set()
    sections: list[SearchSection] = []
    for chunk in chunks:
        section = chunk_to_section[(chunk.document_id, chunk.chunk_id)]
        section_key = chunk_to_section_key[(chunk.document_id, chunk.chunk_id)]
        if section_key in seen:
            continue
        seen.add(section_key)
        sections.append(section)
    return sections


def section_from_chunks(chunks: Sequence[SearchChunk]) -> SearchSection:
    """Build an answer-ready section from one or more chunks."""

    center = max(chunks, key=lambda chunk: chunk.score)
    return SearchSection(
        document_id=center.document_id,
        title=center.title,
        text="\n".join(
            chunk.text for chunk in sorted(chunks, key=lambda c: c.chunk_id)
        ),
        score=max(chunk.score for chunk in chunks),
        url=center.url,
    )


def run_demo() -> list[SearchSection]:
    """Run a tiny end-to-end demo without external services."""

    index = InMemorySearchIndex(
        [
            SearchDocument(
                id="public-guide",
                title="Dense Retrieval Guide",
                contents="Dense retrieval uses embeddings for semantic search.",
                source_type="file",
                document_set="docs",
                tags=frozenset({"retrieval", "public"}),
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            SearchDocument(
                id="private-runbook",
                title="Private Rerank Runbook",
                contents="Rerank deployments require admin approval.",
                source_type="runbook",
                document_set="ops",
                tags=frozenset({"rerank"}),
                updated_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                access=AccessPolicy(
                    public=False,
                    allowed_group_ids=frozenset({"search-admins"}),
                ),
            ),
        ]
    )
    request = SearchRequest(
        query="dense retrieval rerank",
        filters=SearchFilters(tags=frozenset({"retrieval"})),
        limit=3,
    )
    user = SearchUser(id="u1", email="reader@example.test")
    return search_pipeline(request=request, index=index, user=user)


def _assign_section(
    group: Sequence[SearchChunk],
    original_index: dict[tuple[str, int], int],
    chunk_to_section: dict[tuple[str, int], SearchSection],
    chunk_to_section_key: dict[tuple[str, int], tuple[str, int]],
) -> None:
    ranked_group = sorted(
        group,
        key=lambda chunk: original_index.get((chunk.document_id, chunk.chunk_id), 0),
    )
    section = section_from_chunks(ranked_group)
    section_key = (
        section.document_id,
        min(chunk.chunk_id for chunk in ranked_group),
    )
    for chunk in ranked_group:
        chunk_to_section[(chunk.document_id, chunk.chunk_id)] = section
        chunk_to_section_key[(chunk.document_id, chunk.chunk_id)] = section_key


def _matches_filters(document: SearchDocument, filters: IndexFilters) -> bool:
    if filters.attached_document_ids and document.id in filters.attached_document_ids:
        return True
    if filters.source_type and document.source_type != filters.source_type:
        return False
    if (
        filters.document_set is not None
        and document.document_set not in filters.document_set
    ):
        return False
    if filters.tags and not filters.tags.issubset(document.tags):
        return False
    if (
        filters.time_cutoff is not None
        and document.updated_at is not None
        and document.updated_at < filters.time_cutoff
    ):
        return False
    return True


def _score_text(query_terms: set[str], text: str) -> float:
    text_terms = _tokenize(text)
    overlap = query_terms & text_terms
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_terms), 1)


def _split_text(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    for start in range(0, len(text), chunk_size):
        chunks.append(text[start : start + chunk_size])
    return chunks


def _tokenize(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'")
    }


if __name__ == "__main__":
    for result in run_demo():
        print(f"{result.title}: {result.text}")
