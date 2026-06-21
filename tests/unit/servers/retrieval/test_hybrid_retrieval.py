import numpy as np

from src.internal.servers.retrieval.hybrid import DenseEmbeddingRetriever


def _stub_encoder(vectors_by_text):
    """Return a deterministic encoder mapping known texts to unit vectors."""

    def encode(texts):
        rows = []
        for t in texts:
            vec = np.array(vectors_by_text[t], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            rows.append(vec)
        return np.stack(rows)

    return encode


def test_dense_retriever_ranks_by_dot_product():
    docs = [
        {"id": "a", "title": "Cats", "text": "feline animals"},
        {"id": "b", "title": "Dogs", "text": "canine animals"},
    ]
    vecs = {
        "passage: Cats feline animals": [1.0, 0.0],
        "passage: Dogs canine animals": [0.0, 1.0],
        "query: tell me about cats": [0.9, 0.1],
    }
    dense = DenseEmbeddingRetriever(docs, encoder=_stub_encoder(vecs))
    rows = dense.retrieve(["tell me about cats"], topk=2)
    assert [item["document"]["id"] for item in rows[0]] == ["a", "b"]
    assert rows[0][0]["score"] > rows[0][1]["score"]


def test_dense_retriever_empty_corpus_returns_empty_rows():
    dense = DenseEmbeddingRetriever(
        [], encoder=lambda texts: np.empty((0, 0), dtype=np.float32)
    )
    assert dense.retrieve(["anything"], topk=3) == [[]]
