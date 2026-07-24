from src.internal.connectors.models import Document
from src.internal.document_index import chunking
from src.internal.document_index.embedding import deterministic_embedding_fn
from src.internal.document_index.models import ChunkingConfig

EMB = deterministic_embedding_fn(dim=8)


def _doc(text):
    return Document(
        id="d1", title="", contents=text, url=None, metadata={}, permissions=[]
    )


def _semantic(text, chunk_size=900, overlap=0, embedding_fn=EMB, pct=95.0, buf=1):
    return chunking._split_text_semantic(
        text, chunk_size, overlap, embedding_fn, pct, buf
    )


def test_boundary_at_topic_shift():
    # three identical "cats" sentences then two identical "dogs" sentences:
    # within-topic distance ~0, one large spike at the cats->dogs seam.
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    chunks = _semantic(text)
    assert len(chunks) == 2
    assert "cats" in chunks[0] and "dogs" not in chunks[0]
    assert "dogs" in chunks[1] and "cats" not in chunks[1]


def test_flat_document_stays_one_chunk():
    # all-identical sentences → all distances equal → strict > yields no boundary.
    text = "same same same. same same same. same same same."
    chunks = _semantic(text)
    assert len(chunks) == 1


def test_size_cap_never_exceeds_chunk_size():
    # one topic, many sentences, tiny chunk_size → size cap must re-split.
    text = " ".join(["alpha beta gamma delta." for _ in range(30)])
    chunks = _semantic(text, chunk_size=5, overlap=1)
    assert chunks
    for c in chunks:
        assert chunking._token_count(c) <= 5


def test_fallback_when_no_embedder():
    text = "one one one. two two two. three three three."
    assert _semantic(text, embedding_fn=None) == chunking._split_text_paragraphs(
        text, 900, 0
    )


def test_fallback_single_sentence():
    text = "only one sentence here."
    assert _semantic(text) == chunking._split_text_paragraphs(text, 900, 0)


def test_fallback_on_embedder_error():
    def boom(_):
        raise RuntimeError("embedder down")

    text = "one one one. two two two. three three three."
    assert _semantic(text, embedding_fn=boom) == chunking._split_text_paragraphs(
        text, 900, 0
    )


def test_chunk_document_routes_to_semantic_when_enabled():
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    cfg = ChunkingConfig(
        semantic_chunking=True, include_title=False, include_metadata=False
    )
    chunks = chunking.chunk_document(_doc(text), cfg, embedding_fn=EMB)
    assert len(chunks) == 2


def test_chunk_document_semantic_off_matches_today():
    text = "cats cats cats. dogs dogs dogs. birds birds birds."
    cfg = ChunkingConfig(include_title=False, include_metadata=False)  # off
    with_fn = chunking.chunk_document(_doc(text), cfg, embedding_fn=EMB)
    without_fn = chunking.chunk_document(_doc(text), cfg)
    assert [c.text for c in with_fn] == [c.text for c in without_fn]


def test_chunk_documents_threads_embedding_fn():
    text = (
        "cats cats cats. cats cats cats. cats cats cats. "
        "dogs dogs dogs. dogs dogs dogs."
    )
    cfg = ChunkingConfig(
        semantic_chunking=True, include_title=False, include_metadata=False
    )
    chunks = chunking.chunk_documents([_doc(text)], cfg, embedding_fn=EMB)
    assert len(chunks) == 2
