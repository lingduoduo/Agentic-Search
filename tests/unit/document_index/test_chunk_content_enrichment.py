from src.retrieval.models import InferenceChunkUncleaned
from src.backend.document_index.chunk_content_enrichment import (
    cleanup_content_for_chunks,
)


def test_cleanup_strips_title():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="My Title\n\nActual content here.",
        title="My Title",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert len(result) == 1
    assert "My Title" not in result[0].content
    assert "Actual content here." in result[0].content


def test_cleanup_strips_metadata_suffix():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="Main content\n\ntag:value",
        metadata_suffix="tag:value",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert "tag:value" not in result[0].content


def test_cleanup_no_title_noop():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="Just content",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert result[0].content == "Just content"


def test_cleanup_returns_inference_chunk_type():
    from src.retrieval.models import InferenceChunk

    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="content",
    )
    result = cleanup_content_for_chunks([chunk])
    assert isinstance(result[0], InferenceChunk)
