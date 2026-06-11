from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from src.internal.configs.constants import DocumentSource
from src.internal.servers.indexing.models import BaseChunk
from src.internal.servers.indexing.models import IndexingSetting

WEB_SEARCH_PREFIX = "INTERNET_SEARCH_DOC_"


class _SearchSettings:
    """Stub — full model lives in src.internal.db.models (requires SQLAlchemy)."""


SearchSettings = _SearchSettings


class QueryExpansions(BaseModel):
    keywords_expansions: list[str] | None = None
    semantic_expansions: list[str] | None = None


class QueryExpansionType(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"


class SearchSettingsCreationRequest(IndexingSetting):
    @classmethod
    def from_db_model(cls, search_settings: Any) -> "SearchSettingsCreationRequest":
        raise NotImplementedError(
            "SearchSettingsCreationRequest.from_db_model requires a DB-backed SearchSettings"
        )


class SavedSearchSettings(IndexingSetting):
    @classmethod
    def from_db_model(cls, search_settings: Any) -> "SavedSearchSettings":
        raise NotImplementedError(
            "SavedSearchSettings.from_db_model requires a DB-backed SearchSettings"
        )


class Tag(BaseModel):
    tag_key: str
    tag_value: str


class BaseFilters(BaseModel):
    source_type: list[DocumentSource] | None = None
    document_set: list[str] | None = None
    time_cutoff: datetime | None = None
    tags: list[Tag] | None = None


class UserFileFilters(BaseModel):
    project_id_filter: int | None = None
    persona_id_filter: int | None = None


class AssistantKnowledgeFilters(BaseModel):
    """Filters for knowledge attached to an assistant (persona)."""

    attached_document_ids: list[str] | None = None
    hierarchy_node_ids: list[int] | None = None


class IndexFilters(BaseFilters, UserFileFilters, AssistantKnowledgeFilters):
    access_control_list: list[str] | None
    tenant_id: str | None = None


class BasicChunkRequest(BaseModel):
    query: str
    hybrid_alpha: float | None = None
    recency_bias_multiplier: float = 1.0
    limit: int | None = None


class ChunkSearchRequest(BasicChunkRequest):
    user_selected_filters: BaseFilters | None = None
    bypass_acl: bool = False


class ChunkIndexRequest(BasicChunkRequest):
    filters: IndexFilters
    query_keywords: list[str] | None = None


class ContextExpansionType(str, Enum):
    NOT_RELEVANT = "not_relevant"
    MAIN_SECTION_ONLY = "main_section_only"
    INCLUDE_ADJACENT_SECTIONS = "include_adjacent_sections"
    FULL_DOCUMENT = "full_document"


class InferenceChunk(BaseChunk):
    document_id: str
    source_type: DocumentSource
    semantic_identifier: str
    title: str | None
    boost: int
    score: float | None
    hidden: bool
    is_relevant: bool | None = None
    relevance_explanation: str | None = None
    metadata: dict[str, str | list[str]]
    match_highlights: list[str]
    doc_summary: str
    chunk_context: str
    updated_at: datetime | None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    large_chunk_reference_ids: list[int] = Field(default_factory=list)
    is_federated: bool = False
    file_id: str | None = None

    @property
    def unique_id(self) -> str:
        return f"{self.document_id}__{self.chunk_id}"

    def __repr__(self) -> str:
        blurb_words = self.blurb.split()
        short_blurb = ""
        for word in blurb_words:
            if not short_blurb:
                short_blurb = word
                continue
            if len(short_blurb) > 25:
                break
            short_blurb += " " + word
        return f"Inference Chunk: {self.document_id} - {short_blurb}..."

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return False
        return (self.document_id, self.chunk_id) == (other.document_id, other.chunk_id)

    def __hash__(self) -> int:
        return hash((self.document_id, self.chunk_id))

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return NotImplemented
        if self.score is None:
            if other.score is None:
                return self.chunk_id > other.chunk_id
            return True
        if other.score is None:
            return False
        if self.score == other.score:
            return self.chunk_id > other.chunk_id
        return self.score < other.score

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return NotImplemented
        if self.score is None:
            return False
        if other.score is None:
            return True
        if self.score == other.score:
            return self.chunk_id < other.chunk_id
        return self.score > other.score


class InferenceChunkUncleaned(InferenceChunk):
    metadata_suffix: str | None

    def to_inference_chunk(self) -> InferenceChunk:
        inference_chunk_data = {
            k: v for k, v in self.model_dump().items() if k not in ["metadata_suffix"]
        }
        return InferenceChunk(**inference_chunk_data)


class InferenceSection(BaseModel):
    """Section list of chunks with a combined content."""

    center_chunk: InferenceChunk
    chunks: list[InferenceChunk]
    combined_content: str


class SearchDoc(BaseModel):
    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None = None
    blurb: str
    source_type: DocumentSource
    boost: int
    hidden: bool
    metadata: dict[str, str | list[str]]
    score: float | None = None
    is_relevant: bool | None = None
    relevance_explanation: str | None = None
    match_highlights: list[str]
    updated_at: datetime | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    is_internet: bool = False
    file_id: str | None = None

    @classmethod
    def from_chunks_or_sections(
        cls,
        items: "Sequence[InferenceChunk | InferenceSection] | None",
    ) -> list["SearchDoc"]:
        if not items:
            return []

        search_docs = [
            cls(
                document_id=(
                    chunk := (
                        item.center_chunk
                        if isinstance(item, InferenceSection)
                        else item
                    )
                ).document_id,
                chunk_ind=chunk.chunk_id,
                semantic_identifier=chunk.semantic_identifier or "Unknown",
                link=chunk.source_links[0] if chunk.source_links else None,
                blurb=chunk.blurb,
                source_type=chunk.source_type,
                boost=chunk.boost,
                hidden=chunk.hidden,
                metadata=chunk.metadata,
                score=chunk.score,
                match_highlights=chunk.match_highlights,
                updated_at=chunk.updated_at,
                primary_owners=chunk.primary_owners,
                secondary_owners=chunk.secondary_owners,
                is_internet=False,
                file_id=chunk.file_id,
            )
            for item in items
        ]

        return search_docs  # ty: ignore[invalid-return-type]

    @classmethod
    def from_saved_search_doc(cls, saved_search_doc: "SavedSearchDoc") -> "SearchDoc":
        saved_search_doc_data = saved_search_doc.model_dump()
        saved_search_doc_data.pop("db_doc_id", None)
        return cls(**saved_search_doc_data)

    @classmethod
    def from_saved_search_docs(
        cls, saved_search_docs: list["SavedSearchDoc"]
    ) -> list["SearchDoc"]:
        return [
            cls.from_saved_search_doc(saved_search_doc)
            for saved_search_doc in saved_search_docs
        ]

    def model_dump(  # ty: ignore[invalid-method-override]
        self, *args: list, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        initial_dict = super().model_dump(
            *args,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        initial_dict["updated_at"] = (
            self.updated_at.isoformat() if self.updated_at else None
        )
        return initial_dict


class SearchDocsResponse(BaseModel):
    search_docs: list[SearchDoc]
    citation_mapping: dict[int, str]
    displayed_docs: list[SearchDoc] | None = None

    @field_validator("displayed_docs", mode="before")
    @classmethod
    def normalize_empty_displayed_docs(
        cls,
        value: list[SearchDoc] | None,
    ) -> list[SearchDoc] | None:
        return value or None


class SavedSearchDoc(SearchDoc):
    db_doc_id: int
    score: float | None = 0.0

    @classmethod
    def from_search_doc(
        cls, search_doc: SearchDoc, db_doc_id: int = 0
    ) -> "SavedSearchDoc":
        search_doc_data = search_doc.model_dump()
        search_doc_data["score"] = search_doc_data.get("score") or 0.0
        return cls(**search_doc_data, db_doc_id=db_doc_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedSearchDoc":
        return cls(**data)

    @classmethod
    def from_url(cls, url: str) -> "SavedSearchDoc":
        return cls(
            db_doc_id=0,
            document_id=WEB_SEARCH_PREFIX + url,
            chunk_ind=0,
            semantic_identifier=url,
            link=url,
            blurb="",
            source_type=DocumentSource.WEB,
            boost=1,
            hidden=False,
            metadata={},
            score=0.0,
            is_relevant=None,
            relevance_explanation=None,
            match_highlights=[],
            updated_at=None,
            primary_owners=None,
            secondary_owners=None,
            is_internet=True,
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SavedSearchDoc):
            return NotImplemented
        self_score = self.score if self.score is not None else 0.0
        other_score = other.score if other.score is not None else 0.0
        return self_score < other_score


class SavedSearchDocWithContent(SavedSearchDoc):
    content: str


class PersonaSearchInfo(BaseModel):
    """Snapshot of persona data needed by the search pipeline."""

    document_set_names: list[str]
    search_start_date: datetime | None
    attached_document_ids: list[str]
    hierarchy_node_ids: list[int]
