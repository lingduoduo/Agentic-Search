"""Graph RAG: entity-driven retrieval expansion.

Augments standard vector/BM25 retrieval with a lightweight knowledge-graph pass:
  1. Run an initial search to get top-k seed documents.
  2. Extract named entities and acronyms from those documents.
  3. Run additional searches using the top entities as queries.
  4. Fuse all result sets with RRF to produce a final ranked list.

This provides "one-hop graph traversal" semantics without a graph database:
documents linked by shared entities surface together even when neither appears
in the initial result set.

Usage:
    from src.internal.retrieval.graph_rag import graph_rag_search
    results = graph_rag_search("What is dense retrieval?", service=svc, top_k=10)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .backends.base import RetrievalResult
from .fusion import rrf_fuse
from .service import RetrievalService

# Regex for multi-word capitalized phrases (e.g., "Dense Retrieval", "BM25 Ranking")
# and standalone acronyms (e.g., "FAISS", "RAG", "NLP").
_MULTI_WORD_CAP = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]+\b")

_MIN_ENTITY_LEN = 2  # skip single-char acronyms like "I"
_STOPWORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Here",
        "When",
        "Where",
        "Which",
        "What",
        "How",
        "And",
        "But",
        "For",
        "With",
        "From",
        "Into",
        "Over",
        "Also",
        "Each",
        "Both",
    }
)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[str]:
    """Extract named entities and acronyms from a text snippet.

    Returns a deduplicated list preserving first-seen order.
    """
    candidates: list[str] = []
    for m in _MULTI_WORD_CAP.finditer(text):
        candidates.append(m.group())
    for m in _ACRONYM.finditer(text):
        tok = m.group()
        if len(tok) >= _MIN_ENTITY_LEN and tok not in _STOPWORDS:
            candidates.append(tok)

    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Entity graph (in-memory, built from a result batch)
# ---------------------------------------------------------------------------


@dataclass
class EntityGraph:
    """Lightweight co-occurrence graph built from a batch of retrieval results.

    Useful for offline analysis (e.g., inspecting which entities cluster together)
    or as a pre-filtering step before issuing expansion queries.
    """

    doc_entities: dict[str, set[str]] = field(default_factory=dict)
    entity_docs: dict[str, set[str]] = field(default_factory=dict)
    entity_cooccur: dict[str, set[str]] = field(default_factory=dict)

    def add_document(self, doc_id: str, entities: list[str]) -> None:
        self.doc_entities.setdefault(doc_id, set()).update(entities)
        for entity in entities:
            self.entity_docs.setdefault(entity, set()).add(doc_id)
            for other in entities:
                if other != entity:
                    self.entity_cooccur.setdefault(entity, set()).add(other)

    def neighbors(self, entity: str) -> set[str]:
        """Return entities that co-occur with *entity* in any document."""
        return self.entity_cooccur.get(entity, set())

    def docs_for_entity(self, entity: str) -> set[str]:
        """Return all doc_ids that mention *entity*."""
        return self.entity_docs.get(entity, set())


def build_entity_graph(results: list[RetrievalResult]) -> EntityGraph:
    """Build an EntityGraph from a list of retrieval results."""
    graph = EntityGraph()
    for r in results:
        entities = extract_entities(r.title + ". " + r.text)
        graph.add_document(r.doc_id, entities)
    return graph


# ---------------------------------------------------------------------------
# Graph RAG search
# ---------------------------------------------------------------------------


def graph_rag_search(
    query: str,
    *,
    service: RetrievalService,
    top_k: int = 10,
    initial_k: int = 5,
    max_entity_queries: int = 3,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Graph-augmented retrieval via entity-driven query expansion.

    Steps:
      1. Seed search: retrieve *initial_k* docs for *query*.
      2. Extract entities from seed docs.
      3. Run up to *max_entity_queries* additional searches, one per top entity.
      4. Fuse all result lists with RRF and return the top *top_k* docs.

    Falls back to the initial results if expansion produces no new documents.
    """
    seed_results, _ = service.search(query, top_k=initial_k)
    if not seed_results:
        return []

    all_entities: list[str] = []
    for r in seed_results:
        for e in extract_entities(r.title + ". " + r.text):
            if e not in all_entities:
                all_entities.append(e)

    expansion_lists: list[list[RetrievalResult]] = []
    for entity in all_entities[:max_entity_queries]:
        try:
            exp_results, _ = service.search(entity, top_k=top_k)
            if exp_results:
                expansion_lists.append(exp_results)
        except Exception:
            continue

    if not expansion_lists:
        return seed_results[:top_k]

    fused = rrf_fuse([seed_results] + expansion_lists, rrf_k=rrf_k)
    return fused[:top_k]
