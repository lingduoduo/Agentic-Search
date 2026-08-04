"""The bundled retrieval servers honour the access_acl they are sent.

Defense in depth. The web layer enforces regardless, because a third-party
backend is free to ignore the field — but the servers we ship should not.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.servers.retrieval import hybrid
from src.internal.servers.retrieval.demo import TfidfRetriever, create_app

DOCS = [
    {
        "id": "open",
        "title": "Zebra Handbook",
        "contents": "zebra migration",
        "metadata": {"acl": ["public"]},
    },
    {
        "id": "theirs",
        "title": "Zebra Handbook",
        "contents": "zebra migration secrets",
        "metadata": {"acl": ["user:someone_else"]},
    },
    {"id": "undeclared", "title": "Zebra Notes", "contents": "zebra migration notes"},
]


def _client():
    return TestClient(create_app(TfidfRetriever.from_docs(list(DOCS))))


def _ids(payload):
    rows = payload["results"]
    row = rows[0] if rows and isinstance(rows[0], list) else rows
    return {d.get("id") for d in row}


def test_documents_outside_the_acl_are_withheld():
    r = _client().post(
        "/retrieve",
        json={
            "queries": ["zebra migration"],
            "topk": 5,
            "filters": {"access_acl": ["public"]},
        },
    )
    assert r.status_code == 200
    ids = _ids(r.json())
    assert "theirs" not in ids
    assert "open" in ids


def test_documents_with_no_declared_acl_stay_public():
    r = _client().post(
        "/retrieve",
        json={
            "queries": ["zebra migration"],
            "topk": 5,
            "filters": {"access_acl": ["public"]},
        },
    )
    assert "undeclared" in _ids(r.json())


def test_no_filters_returns_everything():
    r = _client().post("/retrieve", json={"queries": ["zebra migration"], "topk": 5})
    assert {"open", "theirs", "undeclared"} <= _ids(r.json())


def test_the_matching_user_sees_their_own_document():
    r = _client().post(
        "/retrieve",
        json={
            "queries": ["zebra migration"],
            "topk": 5,
            "filters": {"access_acl": ["public", "user:someone_else"]},
        },
    )
    assert "theirs" in _ids(r.json())


def _hybrid_client():
    # dense=None (sparse-only) so this doesn't need the e5 model download.
    app = hybrid.create_app(dense=None, sparse=TfidfRetriever.from_docs(list(DOCS)))
    return TestClient(app)


def test_hybrid_server_also_withholds_documents_outside_the_acl():
    r = _hybrid_client().post(
        "/retrieve",
        json={
            "queries": ["zebra migration"],
            "topk": 5,
            "filters": {"access_acl": ["public"]},
        },
    )
    assert r.status_code == 200
    ids = _ids(r.json())
    assert "theirs" not in ids
    assert "open" in ids
    assert "undeclared" in ids
