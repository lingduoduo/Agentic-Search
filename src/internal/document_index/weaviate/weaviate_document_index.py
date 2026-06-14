"""Weaviate implementation of DocumentIndex.

Uses the weaviate-client v4 SDK.  Requires a running Weaviate instance
(HTTP on WEAVIATE_HOST:WEAVIATE_PORT, defaults to localhost:8080).

Collection layout
-----------------
One Weaviate collection per index_name.  Each stored object is a single
document *chunk*.  Vectors are provided by the caller (no Weaviate
vectorizer).  Keyword search uses Weaviate's built-in BM25.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from src.internal.document_index.interfaces import (
    DocumentIndex,
    DocumentInsertionRecord,
    DocumentSectionRequest,
    IndexingMetadata,
    MetadataUpdateRequest,
    TenantState,
)
from src.internal.document_index.models import (
    DocMetadataAwareIndexChunk,
    EmbeddingPrecision,
    Embedding,
    IndexFilters,
    InferenceChunk,
    QueryType,
)
from src.internal.document_index.weaviate.schema import (
    PROP_ACCESS_CONTROL_LIST,
    PROP_BLURB,
    PROP_BOOST,
    PROP_CHUNK_IND,
    PROP_CONTENT,
    PROP_DOCUMENT_ID,
    PROP_DOCUMENT_SETS,
    PROP_HIDDEN,
    PROP_IS_PUBLIC,
    PROP_LARGE_CHUNK_ID,
    PROP_METADATA,
    PROP_SEMANTIC_IDENTIFIER,
    PROP_SOURCE_LINKS,
    PROP_SOURCE_TYPE,
    PROP_TITLE,
    RETRIEVAL_PROPS,
    build_collection_config,
)

if TYPE_CHECKING:
    import weaviate as _weaviate

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class WeaviateDocumentIndex(DocumentIndex):
    """DocumentIndex backed by Weaviate v4.

    Pass ``client`` to inject a pre-built weaviate.WeaviateClient (useful
    for testing).  When omitted the client is created lazily from env vars.
    """

    def __init__(
        self,
        tenant_state: TenantState,
        index_name: str,
        embedding_dim: int = 768,
        embedding_precision: EmbeddingPrecision = EmbeddingPrecision.FLOAT,
        client: "_weaviate.WeaviateClient | None" = None,
        weaviate_host: str = "localhost",
        weaviate_port: int = 8080,
    ) -> None:
        self._tenant_state = tenant_state
        self._collection_name = index_name
        self._embedding_dim = embedding_dim
        self._embedding_precision = embedding_precision
        self._weaviate_host = weaviate_host
        self._weaviate_port = weaviate_port
        self._client = client

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> "_weaviate.WeaviateClient":
        if self._client is None:
            import weaviate

            self._client = weaviate.connect_to_local(
                host=self._weaviate_host,
                port=self._weaviate_port,
            )
        return self._client

    def _get_collection(self) -> Any:
        """Return a Weaviate collection handle, scoped to tenant when needed."""
        col = self._get_client().collections.get(self._collection_name)
        if self._tenant_state.multitenant:
            return col.with_tenant(self._tenant_state.tenant_id)
        return col

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def verify_and_create_index_if_necessary(
        self,
        embedding_dim: int,
        embedding_precision: EmbeddingPrecision,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._embedding_precision = embedding_precision
        client = self._get_client()
        if not client.collections.exists(self._collection_name):
            config = build_collection_config(
                self._collection_name,
                multitenant=self._tenant_state.multitenant,
            )
            client.collections.create(**config)
            logger.info("Created Weaviate collection %r", self._collection_name)
        else:
            logger.debug("Weaviate collection %r already exists", self._collection_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        chunks: Iterable[DocMetadataAwareIndexChunk],
        indexing_metadata: IndexingMetadata,
    ) -> list[DocumentInsertionRecord]:
        collection = self._get_collection()

        # materialise so we can iterate twice (existence check + insert)
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        # Determine which document_ids already have chunks in the index.
        unique_doc_ids = {c.embedded_chunk.chunk.document_id for c in chunk_list}
        already_existed: set[str] = set()
        for doc_id in unique_doc_ids:
            from weaviate.classes.query import Filter

            resp = collection.query.fetch_objects(
                filters=Filter.by_property(PROP_DOCUMENT_ID).equal(doc_id),
                limit=1,
            )
            if resp.objects:
                already_existed.add(doc_id)

        # Delete excess tail chunks for documents that got shorter.
        for doc_id, counts in indexing_metadata.doc_id_to_chunk_cnt_diff.items():
            if counts.old_chunk_cnt > counts.new_chunk_cnt:
                from weaviate.classes.query import Filter

                collection.data.delete_many(
                    where=Filter.all_of(
                        [
                            Filter.by_property(PROP_DOCUMENT_ID).equal(doc_id),
                            Filter.by_property(PROP_CHUNK_IND).greater_or_equal(
                                counts.new_chunk_cnt
                            ),
                        ]
                    )
                )

        # Upsert: delete existing chunk objects then re-insert.
        # (Weaviate v4 has no native upsert; delete-then-insert is idiomatic.)
        for doc_id in already_existed:
            from weaviate.classes.query import Filter

            collection.data.delete_many(
                where=Filter.by_property(PROP_DOCUMENT_ID).equal(doc_id)
            )

        # Build insert objects.
        objects: list[dict] = []
        for chunk in chunk_list:
            ec = chunk.embedded_chunk
            ic = ec.chunk
            acl = _build_acl(chunk)
            obj = {
                "properties": {
                    PROP_DOCUMENT_ID: ic.document_id,
                    PROP_CHUNK_IND: ic.chunk_id,
                    PROP_CONTENT: ic.text,
                    PROP_BLURB: ic.blurb,
                    PROP_TITLE: ic.title or "",
                    PROP_SEMANTIC_IDENTIFIER: ic.title or ic.document_id,
                    PROP_SOURCE_TYPE: "",
                    PROP_SOURCE_LINKS: json.dumps(ic.source_links or {}),
                    PROP_METADATA: json.dumps(ic.metadata or {}),
                    PROP_BOOST: chunk.boost,
                    PROP_HIDDEN: False,
                    PROP_IS_PUBLIC: chunk.access.is_public,
                    PROP_DOCUMENT_SETS: list(chunk.document_sets),
                    PROP_ACCESS_CONTROL_LIST: acl,
                    PROP_LARGE_CHUNK_ID: ic.large_chunk_id,
                },
                "vector": ec.embedding.tolist(),
            }
            objects.append(obj)

        # Batch insert.
        with collection.batch.dynamic() as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj["properties"],
                    vector=obj["vector"],
                )

        return [
            DocumentInsertionRecord(
                document_id=doc_id,
                already_existed=doc_id in already_existed,
            )
            for doc_id in unique_doc_ids
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        document_id: str,
        chunk_count: int | None = None,
    ) -> int:
        from weaviate.classes.query import Filter

        collection = self._get_collection()

        # Count existing chunks before deletion.
        resp = collection.query.fetch_objects(
            filters=Filter.by_property(PROP_DOCUMENT_ID).equal(document_id),
            limit=10_000,
        )
        count = len(resp.objects)
        if count > 0:
            collection.data.delete_many(
                where=Filter.by_property(PROP_DOCUMENT_ID).equal(document_id)
            )
        return count

    # ------------------------------------------------------------------
    # Update metadata
    # ------------------------------------------------------------------

    def update(self, update_requests: list[MetadataUpdateRequest]) -> None:
        from weaviate.classes.query import Filter

        collection = self._get_collection()
        for req in update_requests:
            for doc_id in req.document_ids:
                resp = collection.query.fetch_objects(
                    filters=Filter.by_property(PROP_DOCUMENT_ID).equal(doc_id),
                    limit=10_000,
                )
                for obj in resp.objects:
                    patch: dict[str, Any] = {}
                    if req.hidden is not None:
                        patch[PROP_HIDDEN] = req.hidden
                    if req.boost is not None:
                        patch[PROP_BOOST] = int(req.boost)
                    if req.access is not None:
                        patch[PROP_IS_PUBLIC] = req.access.is_public
                        patch[PROP_ACCESS_CONTROL_LIST] = _acl_from_access(req.access)
                    if req.document_sets is not None:
                        patch[PROP_DOCUMENT_SETS] = list(req.document_sets)
                    if patch:
                        collection.data.update(
                            uuid=obj.uuid,
                            properties=patch,
                        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        query_type: QueryType,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        from weaviate.classes.query import HybridFusion, MetadataQuery

        collection = self._get_collection()
        weaviate_filter = _build_filter(filters)
        resp = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            limit=num_to_retrieve,
            alpha=0.5,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
            return_properties=RETRIEVAL_PROPS,
        )
        return _objects_to_inference_chunks(resp.objects)

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        include_hidden: bool = False,
    ) -> list[InferenceChunk]:
        from weaviate.classes.query import MetadataQuery

        collection = self._get_collection()
        weaviate_filter = _build_filter(filters, include_hidden=include_hidden)
        resp = collection.query.bm25(
            query=query,
            limit=num_to_retrieve,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
            return_properties=RETRIEVAL_PROPS,
        )
        return _objects_to_inference_chunks(resp.objects)

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        from weaviate.classes.query import MetadataQuery

        collection = self._get_collection()
        weaviate_filter = _build_filter(filters)
        resp = collection.query.near_vector(
            near_vector=query_embedding,
            limit=num_to_retrieve,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(distance=True, certainty=True),
            return_properties=RETRIEVAL_PROPS,
        )
        return _objects_to_inference_chunks(resp.objects, use_distance=True)

    def id_based_retrieval(
        self,
        chunk_requests: list[DocumentSectionRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
    ) -> list[InferenceChunk]:
        from weaviate.classes.query import Filter

        collection = self._get_collection()
        results: list[InferenceChunk] = []
        for req in chunk_requests:
            conditions = [Filter.by_property(PROP_DOCUMENT_ID).equal(req.document_id)]
            if req.min_chunk_ind is not None:
                conditions.append(
                    Filter.by_property(PROP_CHUNK_IND).greater_or_equal(
                        req.min_chunk_ind
                    )
                )
            if req.max_chunk_ind is not None:
                conditions.append(
                    Filter.by_property(PROP_CHUNK_IND).less_or_equal(req.max_chunk_ind)
                )
            weaviate_filter = (
                Filter.all_of(conditions) if len(conditions) > 1 else conditions[0]
            )
            resp = collection.query.fetch_objects(
                filters=weaviate_filter,
                limit=10_000,
                return_properties=RETRIEVAL_PROPS,
            )
            chunks = _objects_to_inference_chunks(resp.objects)
            chunks.sort(key=lambda c: c.chunk_ind)
            results.extend(chunks)
        return results

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
        dirty: bool | None = None,
    ) -> list[InferenceChunk]:
        collection = self._get_collection()
        weaviate_filter = _build_filter(filters)
        resp = collection.query.fetch_objects(
            limit=num_to_retrieve,
            filters=weaviate_filter,
            return_properties=RETRIEVAL_PROPS,
        )
        return _objects_to_inference_chunks(resp.objects)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_acl(chunk: DocMetadataAwareIndexChunk) -> list[str]:
    return _acl_from_access(chunk.access)


def _acl_from_access(access: Any) -> list[str]:
    from src.internal.configs.constants import PUBLIC_DOC_PAT

    acl: list[str] = list(access.user_ids) + list(access.group_ids)
    if access.is_public:
        acl.append(PUBLIC_DOC_PAT)
    return acl


def _build_filter(
    filters: IndexFilters,
    include_hidden: bool = False,
) -> Any:
    """Convert IndexFilters to a Weaviate Filter tree, or None."""
    from weaviate.classes.query import Filter

    conditions: list[Any] = []

    if not include_hidden:
        conditions.append(Filter.by_property(PROP_HIDDEN).equal(False))

    if filters.access_control_list:
        acl_conditions = [
            Filter.by_property(PROP_IS_PUBLIC).equal(True),
        ] + [
            Filter.by_property(PROP_ACCESS_CONTROL_LIST).contains_any([entry])
            for entry in filters.access_control_list
        ]
        conditions.append(Filter.any_of(acl_conditions))

    if filters.document_set:
        ds_conditions = [
            Filter.by_property(PROP_DOCUMENT_SETS).contains_any([ds])
            for ds in filters.document_set
        ]
        conditions.append(Filter.any_of(ds_conditions))

    if filters.source_type:
        st_conditions = [
            Filter.by_property(PROP_SOURCE_TYPE).equal(st) for st in filters.source_type
        ]
        conditions.append(Filter.any_of(st_conditions))

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return Filter.all_of(conditions)


def _objects_to_inference_chunks(
    objects: list[Any],
    use_distance: bool = False,
) -> list[InferenceChunk]:
    chunks: list[InferenceChunk] = []
    for obj in objects:
        props = obj.properties
        score: float | None = None
        if obj.metadata:
            if use_distance and obj.metadata.distance is not None:
                # Convert distance to similarity: 1 - distance for cosine
                score = 1.0 - float(obj.metadata.distance)
            elif not use_distance and obj.metadata.score is not None:
                score = float(obj.metadata.score)

        source_links: dict | None = None
        raw_links = props.get(PROP_SOURCE_LINKS)
        if raw_links:
            try:
                source_links = json.loads(raw_links)
            except (json.JSONDecodeError, TypeError):
                pass

        metadata: dict = {}
        raw_meta = props.get(PROP_METADATA)
        if raw_meta:
            try:
                metadata = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                pass

        large_chunk_id = props.get(PROP_LARGE_CHUNK_ID)

        chunks.append(
            InferenceChunk(
                document_id=props.get(PROP_DOCUMENT_ID, ""),
                chunk_ind=int(props.get(PROP_CHUNK_IND, 0)),
                blurb=props.get(PROP_BLURB, ""),
                content=props.get(PROP_CONTENT, ""),
                source_links=source_links,
                semantic_identifier=props.get(PROP_SEMANTIC_IDENTIFIER, ""),
                boost=int(props.get(PROP_BOOST, 0)),
                hidden=bool(props.get(PROP_HIDDEN, False)),
                score=score,
                metadata=metadata,
                document_sets=set(props.get(PROP_DOCUMENT_SETS) or []),
                access_control_list=props.get(PROP_ACCESS_CONTROL_LIST),
                title=props.get(PROP_TITLE),
                source_type=props.get(PROP_SOURCE_TYPE, ""),
                large_chunk_id=int(large_chunk_id)
                if large_chunk_id is not None
                else None,
            )
        )
    return chunks
