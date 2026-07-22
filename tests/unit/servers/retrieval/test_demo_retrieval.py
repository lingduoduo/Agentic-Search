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


def test_from_docs_builds_retriever_without_a_file():
    docs = [
        {"id": "a", "title": "Cats", "contents": "feline animals purr"},
        {"id": "b", "title": "Dogs", "contents": "canine animals bark"},
    ]
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["feline purr"], topk=5)
    assert rows[0][0]["document"]["id"] == "a"


def test_retrieve_forwards_document_source_metadata():
    # The /retrieve response must forward metadata (incl. the per-document
    # "source") so downstream source cards show a real origin, not "Unknown".
    docs = [
        {
            "id": "a",
            "title": "Cats",
            "contents": "feline animals purr",
            "metadata": {"source": "Team Wiki"},
        },
    ]
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["feline purr"], topk=5)
    assert rows[0][0]["document"]["metadata"]["source"] == "Team Wiki"
