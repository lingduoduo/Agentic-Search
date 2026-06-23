import json

from src.internal.servers.retrieval.demo import TfidfRetriever


def _write_corpus(tmp_path, docs):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return str(path)


def test_retrieve_ranks_matching_documents(tmp_path):
    docs = [
        {"id": "a", "title": "Cats", "contents": "feline animals purr"},
        {"id": "b", "title": "Dogs", "contents": "canine animals bark"},
    ]
    retriever = TfidfRetriever(_write_corpus(tmp_path, docs))
    rows = retriever.retrieve(["feline purr"], topk=5)
    assert rows[0][0]["document"]["id"] == "a"


def test_retrieve_drops_zero_relevance_documents(tmp_path):
    """A query sharing no terms with the corpus must return no results,
    not arbitrary top-k documents with score 0.0."""
    docs = [
        {"id": "a", "title": "Cats", "contents": "feline animals purr"},
        {"id": "b", "title": "Dogs", "contents": "canine animals bark"},
    ]
    retriever = TfidfRetriever(_write_corpus(tmp_path, docs))
    rows = retriever.retrieve(["GRPO"], topk=5)
    assert rows[0] == []
