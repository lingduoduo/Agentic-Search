from datetime import datetime

from src.retrieval.models import (
    QueryType,
    Embedding,
    InferenceChunk,
    InferenceChunkUncleaned,
    IndexFilters,
    MultipassConfig,
    ExternalAccess,
)


def test_query_type_values():
    assert QueryType.SEMANTIC == "semantic"
    assert QueryType.KEYWORD == "keyword"
    assert QueryType.HYBRID == "hybrid"


def test_embedding_is_list_of_float():
    emb: Embedding = [0.1, 0.2, 0.3]
    assert isinstance(emb, list)


def test_inference_chunk_defaults():
    chunk = InferenceChunk(document_id="doc1", chunk_ind=0)
    assert chunk.blurb == ""
    assert chunk.content == ""
    assert chunk.score is None
    assert chunk.match_highlights == []
    assert chunk.document_sets == set()


def test_inference_chunk_full():
    chunk = InferenceChunk(
        document_id="doc1",
        chunk_ind=2,
        blurb="blurb text",
        content="full content",
        score=0.95,
        source_type="web",
        match_highlights=["highlight"],
    )
    assert chunk.content == "full content"
    assert chunk.score == 0.95


def test_inference_chunk_uncleaned_to_inference_chunk():
    raw = InferenceChunkUncleaned(
        document_id="doc1",
        chunk_ind=0,
        content="Title\n\nReal content",
        title="My Title",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    cleaned = raw.to_inference_chunk()
    assert isinstance(cleaned, InferenceChunk)
    assert cleaned.document_id == "doc1"


def test_index_filters_defaults():
    f = IndexFilters()
    assert f.access_control_list is None
    assert f.document_set is None
    assert f.source_type is None
    assert f.time_cutoff is None


def test_index_filters_with_values():
    f = IndexFilters(
        access_control_list=["user1", "PUBLIC"],
        document_set=["set1"],
        source_type=["web"],
        time_cutoff=datetime(2024, 1, 1),
    )
    assert "PUBLIC" in f.access_control_list
    assert f.document_set == ["set1"]


def test_multipass_config_defaults():
    cfg = MultipassConfig()
    assert cfg.multipass_indexing is False
    assert cfg.enable_large_chunks is False


def test_external_access_defaults():
    ea = ExternalAccess()
    assert ea.external_user_emails == []
    assert ea.external_user_group_ids == []
    assert ea.is_public is False
