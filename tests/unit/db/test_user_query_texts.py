from src.internal.db.store import AgenticSearchStore
from src.internal.db.models import UserRecord


def _seed(store, session_id, pairs):
    store.upsert_user(UserRecord(id="u1"))
    store.create_chat_session(session_id=session_id, user_id="u1")
    for role, content in pairs:
        store.add_chat_message(session_id, role=role, content=content)


def test_returns_distinct_user_queries_newest_first():
    store = AgenticSearchStore(":memory:")
    _seed(
        store,
        "s1",
        [
            ("user", "what is FAISS"),
            ("assistant", "FAISS is a library"),
            ("user", "find the Q3 report"),
            ("user", "what is FAISS"),  # duplicate
        ],
    )
    got = store.get_user_query_texts()
    assert got == ["find the Q3 report", "what is FAISS"]  # distinct, newest first
    assert store.get_user_query_texts(limit=1) == ["find the Q3 report"]


def test_ignores_blank_and_non_user():
    store = AgenticSearchStore(":memory:")
    _seed(store, "s2", [("assistant", "hi"), ("user", "   "), ("user", "HNSW")])
    assert store.get_user_query_texts() == ["HNSW"]
