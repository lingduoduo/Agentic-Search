# tests/unit/db/test_user_profiles.py
from src.internal.db.store import AgenticSearchStore


def test_replace_and_get_profile_overwrites():
    store = AgenticSearchStore(":memory:")
    first = store.replace_user_profile(
        "u1",
        [{"topic": "work", "subtopic": "role", "content": "software engineer"}],
    )
    assert len(first) == 1 and first[0].topic == "work"

    # regeneration fully replaces the prior profile
    store.replace_user_profile(
        "u1",
        [
            {"topic": "home", "subtopic": "city", "content": "Shanghai"},
            {"topic": "food", "subtopic": "", "content": "likes Sichuan"},
        ],
    )
    got = store.get_user_profile("u1")
    assert [e.topic for e in got] == ["food", "home"]  # ordered by topic
    # entries with neither topic nor content are dropped
    store.replace_user_profile("u1", [{"topic": "", "subtopic": "", "content": ""}])
    assert store.get_user_profile("u1") == []
    store.close()
