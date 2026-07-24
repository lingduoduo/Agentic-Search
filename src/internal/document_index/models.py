"""Shared models for retrieval indexing and embedding flows."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from dataclasses import field

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.internal.connectors.models import ConnectorFailure
from src.internal.connectors.models import Document

MAX_METADATA_PERCENTAGE = 0.25
CHUNK_MIN_CONTENT = 16


# ---------------------------------------------------------------------------
# SearchDoc — canonical search result used by chat, citation, and retrieval
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel as _PydanticBase
    from pydantic import Field as _Field

    class SearchDoc(_PydanticBase):
        """A single retrieved document chunk returned by the search layer.

        Used throughout the chat pipeline (citation processor, llm_loop, etc.)
        and the retrieval servers.
        """

        document_id: str
        chunk_ind: int = 0
        semantic_identifier: str = ""
        link: str | None = None
        blurb: str = ""
        source_type: str = ""
        boost: int = 0
        hidden: bool = False
        metadata: dict[str, Any] = _Field(default_factory=dict)
        score: float = 0.0
        match_highlights: list[str] = _Field(default_factory=list)

    class SearchDocsResponse(_PydanticBase):
        """Wrapper returned by search tool calls."""

        search_docs: list[SearchDoc] = _Field(default_factory=list)
        displayed_docs: list[SearchDoc] | None = None

except ImportError:  # pydantic not available (rare — retrieval-only environments)

    @dataclass  # type: ignore[no-redef]
    class SearchDoc:  # type: ignore[no-redef]
        document_id: str
        chunk_ind: int = 0
        semantic_identifier: str = ""
        link: str | None = None
        blurb: str = ""
        source_type: str = ""
        boost: int = 0
        hidden: bool = False
        metadata: dict = field(default_factory=dict)
        score: float = 0.0
        match_highlights: list = field(default_factory=list)

    @dataclass  # type: ignore[no-redef]
    class SearchDocsResponse:  # type: ignore[no-redef]
        search_docs: list = field(default_factory=list)
        displayed_docs: list | None = None


class EmbeddingPrecision(StrEnum):
    FLOAT = "float"
    BFLOAT16 = "bfloat16"


class SwitchoverType(StrEnum):
    REINDEX = "reindex"
    SWAP = "swap"


class EmbeddingProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    CUSTOM = "custom"


class QueryType(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# Type alias matching shared_configs.model_server_models.Embedding
Embedding = list[float]


@dataclass(frozen=True)
class ChunkEmbedding:
    """Full and optional mini-chunk embeddings for one indexed chunk."""

    full_embedding: np.ndarray
    mini_chunk_embeddings: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class BaseChunk:
    """Content-level chunk metadata shared by indexing and retrieval."""

    chunk_id: int
    blurb: str
    content: str
    source_links: dict[int, str] | None = None
    image_file_id: str | None = None
    section_continuation: bool = False


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls document chunking."""

    chunk_size: int = 900
    chunk_overlap: int = 120
    include_title: bool = True
    include_metadata: bool = True
    blurb_size: int = 48
    enable_large_chunks: bool = False
    large_chunk_ratio: int = 4
    enable_mini_chunks: bool = False
    mini_chunk_size: int = 128
    max_document_chars: int | None = None
    max_metadata_percentage: float = MAX_METADATA_PERCENTAGE
    min_content_tokens: int = CHUNK_MIN_CONTENT
    semantic_chunking: bool = False
    semantic_breakpoint_percentile: float = 95.0
    semantic_buffer_size: int = 1

    def validate(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.blurb_size < 1:
            raise ValueError("blurb_size must be at least 1.")
        if self.large_chunk_ratio < 2:
            raise ValueError("large_chunk_ratio must be at least 2.")
        if self.mini_chunk_size < 1:
            raise ValueError("mini_chunk_size must be at least 1.")
        if self.max_document_chars is not None and self.max_document_chars < 1:
            raise ValueError("max_document_chars must be at least 1.")
        if not 0 <= self.max_metadata_percentage <= 1:
            raise ValueError("max_metadata_percentage must be between 0 and 1.")
        if self.min_content_tokens < 1:
            raise ValueError("min_content_tokens must be at least 1.")
        if not 0 < self.semantic_breakpoint_percentile < 100:
            raise ValueError(
                "semantic_breakpoint_percentile must be between 0 and 100 (exclusive)."
            )
        if self.semantic_buffer_size < 1:
            raise ValueError("semantic_buffer_size must be at least 1.")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Controls embedding preparation and batching."""

    retrieval_method: str = "contriever"
    batch_size: int = 64
    normalize_embeddings: bool = False
    query_prefix: str | None = None
    passage_prefix: str | None = None
    embed_titles: bool = False
    isolate_failures: bool = False
    failure_retry_seconds: float = 0.0

    def validate(self) -> None:
        if not self.retrieval_method.strip():
            raise ValueError("retrieval_method is required.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.failure_retry_seconds < 0:
            raise ValueError("failure_retry_seconds must be non-negative.")


@dataclass(frozen=True)
class IndexWriterConfig:
    """Controls files written by the indexing pipeline."""

    save_dir: str | Path
    write_faiss: bool = False
    faiss_type: str = "Flat"
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    corpus_filename: str = "corpus.jsonl"
    embedding_filename: str = "embeddings.memmap"
    index_filename: str = "dense_Flat.index"

    def validate(self) -> None:
        if not str(self.save_dir):
            raise ValueError("save_dir is required.")
        if self.hnsw_ef_construction is not None and self.hnsw_ef_construction < 1:
            raise ValueError("hnsw_ef_construction must be at least 1.")
        if self.hnsw_ef_search is not None and self.hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be at least 1.")


@dataclass(frozen=True)
class IndexingPipelineConfig:
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    writer: IndexWriterConfig

    def validate(self) -> None:
        self.chunking.validate()
        self.embedding.validate()
        self.writer.validate()


@dataclass(frozen=True)
class IndexChunk:
    """One chunk ready for embedding."""

    id: str
    document_id: str
    chunk_id: int
    text: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] | None = None
    blurb: str = ""
    source_links: dict[int, str] | None = None
    image_file_id: str | None = None
    section_continuation: bool = False
    metadata_suffix_semantic: str = ""
    metadata_suffix_keyword: str = ""
    mini_chunk_texts: list[str] | None = None
    large_chunk_reference_ids: list[int] | None = None
    large_chunk_id: int | None = None
    title_prefix: str = ""
    doc_summary: str = ""
    chunk_context: str = ""

    @property
    def content(self) -> str:
        return self.text

    def to_short_descriptor(self) -> str:
        return f"{self.document_id} Chunk ID: {self.chunk_id}"

    def get_link(self) -> str | None:
        return self.url


@dataclass(frozen=True)
class EmbeddedChunk:
    """One chunk plus its embedding vectors."""

    chunk: IndexChunk
    embedding: np.ndarray
    title_embedding: np.ndarray | None = None
    mini_chunk_embeddings: list[np.ndarray] = field(default_factory=list)

    @property
    def embeddings(self) -> ChunkEmbedding:
        return ChunkEmbedding(
            full_embedding=self.embedding,
            mini_chunk_embeddings=self.mini_chunk_embeddings,
        )


@dataclass(frozen=True)
class DocumentAccess:
    """Lightweight document access metadata for search indexing."""

    is_public: bool = True
    user_ids: set[str] = field(default_factory=set)
    group_ids: set[str] = field(default_factory=set)


@dataclass
class ExternalAccess:
    """External (third-party) access control metadata for a document."""

    external_user_emails: list[str] = field(default_factory=list)
    external_user_group_ids: list[str] = field(default_factory=list)
    is_public: bool = False


@dataclass(frozen=True)
class DocMetadataAwareIndexChunk:
    """Embedded chunk plus metadata needed by a search index writer."""

    embedded_chunk: EmbeddedChunk
    tenant_id: str
    access: DocumentAccess
    document_sets: set[str] = field(default_factory=set)
    user_project: list[int] = field(default_factory=list)
    personas: list[int] = field(default_factory=list)
    boost: int = 0
    aggregated_chunk_boost_factor: float = 1.0
    ancestor_hierarchy_node_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_embedded_chunk(
        cls,
        embedded_chunk: EmbeddedChunk,
        *,
        access: DocumentAccess,
        tenant_id: str,
        document_sets: set[str] | None = None,
        user_project: list[int] | None = None,
        personas: list[int] | None = None,
        boost: int = 0,
        aggregated_chunk_boost_factor: float = 1.0,
        ancestor_hierarchy_node_ids: list[int] | None = None,
    ) -> "DocMetadataAwareIndexChunk":
        return cls(
            embedded_chunk=embedded_chunk,
            access=access,
            document_sets=document_sets or set(),
            user_project=user_project or [],
            personas=personas or [],
            boost=boost,
            aggregated_chunk_boost_factor=aggregated_chunk_boost_factor,
            tenant_id=tenant_id,
            ancestor_hierarchy_node_ids=ancestor_hierarchy_node_ids or [],
        )

    @property
    def title_prefix(self) -> str:
        return self.embedded_chunk.chunk.title_prefix

    @property
    def doc_summary(self) -> str:
        return self.embedded_chunk.chunk.doc_summary

    @property
    def chunk_context(self) -> str:
        return self.embedded_chunk.chunk.chunk_context

    @property
    def content(self) -> str:
        return self.embedded_chunk.chunk.text

    @property
    def metadata_suffix_keyword(self) -> str:
        return self.embedded_chunk.chunk.metadata_suffix_keyword

    @property
    def metadata_suffix_semantic(self) -> str:
        return self.embedded_chunk.chunk.metadata_suffix_semantic

    @property
    def chunk_id(self) -> int:
        return self.embedded_chunk.chunk.chunk_id

    @property
    def large_chunk_id(self) -> int | None:
        return self.embedded_chunk.chunk.large_chunk_id


@dataclass(frozen=True)
class IndexingPipelineResult:
    total_documents: int
    total_chunks: int
    corpus_path: Path
    embedding_path: Path
    index_path: Path | None = None
    failures: list[ConnectorFailure] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddingModelDetail:
    """Settings needed to call an embedding model."""

    model_name: str
    normalize: bool
    query_prefix: str | None = None
    passage_prefix: str | None = None
    id: int | None = None
    api_url: str | None = None
    provider_type: EmbeddingProvider | None = None
    api_key: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EmbeddingModelDetail":
        return cls(
            id=data.get("id"),
            model_name=str(data["model_name"]),
            normalize=bool(data.get("normalize", False)),
            query_prefix=data.get("query_prefix"),
            passage_prefix=data.get("passage_prefix"),
            provider_type=data.get("provider_type"),
            api_key=data.get("api_key"),
            api_url=data.get("api_url"),
        )


@dataclass(frozen=True)
class IndexingSetting(EmbeddingModelDetail):
    """Additional embedding and index metadata needed at indexing time."""

    model_dim: int = 0
    index_name: str | None = None
    multipass_indexing: bool = False
    embedding_precision: EmbeddingPrecision = EmbeddingPrecision.FLOAT
    reduced_dimension: int | None = None
    switchover_type: SwitchoverType = SwitchoverType.REINDEX
    enable_contextual_rag: bool = False
    contextual_rag_model_configuration_id: int | None = None
    contextual_rag_llm_name: str | None = None
    contextual_rag_llm_provider: str | None = None

    @property
    def final_embedding_dim(self) -> int:
        return self.reduced_dimension or self.model_dim

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "IndexingSetting":
        return cls(
            id=data.get("id"),
            model_name=str(data["model_name"]),
            model_dim=int(data["model_dim"]),
            normalize=bool(data.get("normalize", False)),
            query_prefix=data.get("query_prefix"),
            passage_prefix=data.get("passage_prefix"),
            provider_type=data.get("provider_type"),
            api_key=data.get("api_key"),
            api_url=data.get("api_url"),
            index_name=data.get("index_name"),
            multipass_indexing=bool(data.get("multipass_indexing", False)),
            embedding_precision=data.get(
                "embedding_precision", EmbeddingPrecision.FLOAT
            ),
            reduced_dimension=data.get("reduced_dimension"),
            switchover_type=data.get("switchover_type", SwitchoverType.REINDEX),
            enable_contextual_rag=bool(data.get("enable_contextual_rag", False)),
            contextual_rag_model_configuration_id=data.get(
                "contextual_rag_model_configuration_id"
            ),
        )


@dataclass(frozen=True)
class MultipassConfig:
    """Configuration for multipass indexing (large + mini chunks)."""

    multipass_indexing: bool = False
    enable_large_chunks: bool = False


@dataclass(frozen=True)
class UpdatableChunkData:
    chunk_id: int
    document_id: str
    boost_score: float


class ChunkEnrichmentContext(Protocol):
    doc_id_to_previous_chunk_cnt: dict[str, int]
    doc_id_to_new_chunk_cnt: dict[str, int]

    def enrich_chunk(
        self,
        chunk: EmbeddedChunk,
        score: float,
    ) -> DocMetadataAwareIndexChunk: ...


class IndexingBatchAdapter(Protocol):
    connector_id: int | None
    credential_id: int | None

    def prepare(
        self,
        documents: list[Document],
        ignore_time_skip: bool,
    ) -> Any | None: ...

    @contextlib.contextmanager
    def lock_context(self, documents: list[Document]) -> Generator[Any, None, None]:
        yield None

    def prepare_enrichment(
        self,
        context: Any,
        tenant_id: str,
        chunks: list[EmbeddedChunk],
        db_session: Any,
    ) -> ChunkEnrichmentContext: ...

    def post_index(
        self,
        context: Any,
        updatable_chunk_data: list[UpdatableChunkData],
        filtered_documents: list[Document],
        enrichment: ChunkEnrichmentContext,
        db_session: Any,
    ) -> None: ...


# ---------------------------------------------------------------------------
# DocumentIndex retrieval types (Pydantic for schema validation)
# ---------------------------------------------------------------------------

try:
    from datetime import datetime as _dt

    class InferenceChunk(_PydanticBase):
        """A retrieved chunk returned from a DocumentIndex retrieval method."""

        document_id: str
        chunk_ind: int
        blurb: str = ""
        content: str = ""
        source_links: dict[int, str] | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        score: float | None = None
        metadata: dict[str, Any] = _Field(default_factory=dict)
        match_highlights: list[str] = _Field(default_factory=list)
        document_sets: set[str] = _Field(default_factory=set)
        access_control_list: list[str] | None = None
        title: str | None = None
        source_type: str = ""
        large_chunk_id: int | None = None
        large_chunk_reference_ids: list[int] | None = None

    class InferenceChunkUncleaned(_PydanticBase):
        """Mutable InferenceChunk used during content cleanup pipeline."""

        document_id: str
        chunk_ind: int
        content: str = ""
        title: str | None = None
        blurb: str = ""
        source_links: dict[int, str] | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        score: float | None = None
        metadata: dict[str, Any] = _Field(default_factory=dict)
        match_highlights: list[str] = _Field(default_factory=list)
        document_sets: set[str] = _Field(default_factory=set)
        access_control_list: list[str] | None = None
        source_type: str = ""
        metadata_suffix: str = ""
        doc_summary: str = ""
        chunk_context: str = ""

        def to_inference_chunk(self) -> "InferenceChunk":
            return InferenceChunk(
                document_id=self.document_id,
                chunk_ind=self.chunk_ind,
                blurb=self.blurb,
                content=self.content,
                source_links=self.source_links,
                section_continuation=self.section_continuation,
                semantic_identifier=self.semantic_identifier,
                boost=self.boost,
                hidden=self.hidden,
                score=self.score,
                metadata=self.metadata,
                match_highlights=self.match_highlights,
                document_sets=self.document_sets,
                access_control_list=self.access_control_list,
                source_type=self.source_type,
            )

    class IndexFilters(_PydanticBase):
        """Filters passed to DocumentIndex retrieval methods."""

        model_config = {"frozen": True}

        access_control_list: list[str] | None = None
        document_set: list[str] | None = None
        source_type: list[str] | None = None
        tags: dict[str, list[str]] | None = None
        time_cutoff: _dt | None = None
        is_public: bool | None = None
        tenant_id: str | None = None

except ImportError:

    @dataclass  # type: ignore[no-redef]
    class InferenceChunk:  # type: ignore[no-redef]
        document_id: str
        chunk_ind: int
        blurb: str = ""
        content: str = ""
        source_links: dict | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        metadata: dict = field(default_factory=dict)
        score: float | None = None
        match_highlights: list = field(default_factory=list)
        document_sets: set = field(default_factory=set)
        access_control_list: list | None = None
        title: str | None = None
        source_type: str = ""
        large_chunk_id: int | None = None
        large_chunk_reference_ids: list | None = None

    @dataclass  # type: ignore[no-redef]
    class InferenceChunkUncleaned:  # type: ignore[no-redef]
        document_id: str
        chunk_ind: int
        content: str = ""
        title: str | None = None
        blurb: str = ""
        source_links: dict | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        metadata: dict = field(default_factory=dict)
        score: float | None = None
        match_highlights: list = field(default_factory=list)
        document_sets: set = field(default_factory=set)
        access_control_list: list | None = None
        source_type: str = ""
        metadata_suffix: str = ""
        doc_summary: str = ""
        chunk_context: str = ""

        def to_inference_chunk(self) -> "InferenceChunk":
            return InferenceChunk(
                document_id=self.document_id,
                chunk_ind=self.chunk_ind,
                blurb=self.blurb,
                content=self.content,
            )

    @dataclass  # type: ignore[no-redef]
    class IndexFilters:  # type: ignore[no-redef]
        access_control_list: list | None = None
        document_set: list | None = None
        source_type: list | None = None
        tags: dict | None = None
        time_cutoff: Any = None
        is_public: bool | None = None
        tenant_id: str | None = None
