from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt, **kwargs):
        return self._text


def test_generate_profile_persists_parsed_entries():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is a software engineer at TechCorp")
    llm = _FakeLLM(
        'Here is the profile: [{"topic": "work", "subtopic": "role", '
        '"content": "Software engineer at TechCorp"}] done'
    )
    entries = service.generate_user_profile(store, "u1", llm)
    assert len(entries) == 1 and entries[0].topic == "work"
    assert (
        service.get_user_profile(store, "u1")[0].content
        == "Software engineer at TechCorp"
    )
    store.close()


def test_generate_profile_malformed_json_yields_empty():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "x")
    entries = service.generate_user_profile(store, "u1", _FakeLLM("not json at all"))
    assert entries == []
    store.close()


def test_generate_profile_no_memories_clears_profile():
    store = AgenticSearchStore(":memory:")
    store.replace_user_profile(
        "u1", [{"topic": "old", "subtopic": "", "content": "stale"}]
    )
    entries = service.generate_user_profile(store, "u1", _FakeLLM("[]"))
    assert entries == []
    assert service.get_user_profile(store, "u1") == []
    store.close()
