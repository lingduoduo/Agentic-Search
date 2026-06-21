import numpy as np
from fastapi.testclient import TestClient

from src.internal.servers.retrieval.hybrid import DenseEmbeddingRetriever, create_app


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


class _FakeRetriever:
    """Returns preconfigured rows per call, shaped like TfidfRetriever.retrieve."""

    def __init__(self, rows_by_query):
        self._rows_by_query = rows_by_query

    def retrieve(self, queries, topk):
        out = []
        for q in queries:
            out.append(self._rows_by_query.get(q, [])[:topk])
        return out


def _doc(doc_id, score):
    return {
        "document": {"id": doc_id, "title": doc_id, "text": "", "url": None},
        "score": score,
    }


def test_hybrid_retrieve_fuses_dense_and_sparse():
    # 'shared' appears in both legs → should outrank single-leg docs after RRF.
    dense = _FakeRetriever({"q": [_doc("shared", 0.9), _doc("dense_only", 0.8)]})
    sparse = _FakeRetriever({"q": [_doc("shared", 0.5), _doc("sparse_only", 0.4)]})
    client = TestClient(create_app(dense=dense, sparse=sparse))
    resp = client.post(
        "/retrieve", json={"query": "q", "topk": 3, "return_scores": True}
    )
    assert resp.status_code == 200
    ids = [item["document"]["id"] for item in resp.json()["results"]]
    assert ids[0] == "shared"
    assert set(ids) == {"shared", "dense_only", "sparse_only"}


def test_hybrid_retrieve_contract_single_and_batch():
    dense = _FakeRetriever({"q1": [_doc("a", 0.9)], "q2": [_doc("b", 0.9)]})
    sparse = _FakeRetriever({"q1": [_doc("a", 0.5)], "q2": [_doc("b", 0.5)]})
    client = TestClient(create_app(dense=dense, sparse=sparse))
    single = client.post("/retrieve", json={"query": "q1", "topk": 2}).json()
    assert single["results"][0]["id"] == "a"
    batch = client.post("/retrieve", json={"queries": ["q1", "q2"], "topk": 2}).json()
    assert [row[0]["id"] for row in batch["results"]] == ["a", "b"]


def test_hybrid_retrieve_degrades_to_sparse_when_dense_none():
    sparse = _FakeRetriever({"q": [_doc("s1", 0.5), _doc("s2", 0.4)]})
    client = TestClient(create_app(dense=None, sparse=sparse))
    resp = client.post(
        "/retrieve", json={"query": "q", "topk": 2, "return_scores": True}
    )
    assert resp.status_code == 200
    assert [item["document"]["id"] for item in resp.json()["results"]] == ["s1", "s2"]
