# tests/unit/memory/test_service_basic.py
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service


def _store():
    return AgenticSearchStore(":memory:")


def test_save_and_lexical_search():
    store = _store()
    service.save_memory(store, "u1", "User enjoys hiking in the mountains")
    service.save_memory(store, "u1", "User is allergic to peanuts")
    hits = service.search_memories(store, "u1", "mountain hiking trip")
    assert hits and "hiking" in hits[0][0].memory_text
    store.close()


def test_search_with_injected_encoder():
    import numpy as np

    store = _store()
    service.save_memory(store, "u1", "alpha")
    service.save_memory(store, "u1", "omega")

    def fake_encoder(texts):
        # "query: alpha" ~ "passage: alpha": map by substring to orthogonal vectors
        vecs = []
        for t in texts:
            vecs.append([1.0, 0.0] if "alpha" in t else [0.0, 1.0])
        return np.array(vecs, dtype=np.float32)

    hits = service.search_memories(store, "u1", "alpha", encoder=fake_encoder)
    assert hits[0][0].memory_text == "alpha"
    store.close()


def test_consolidate_dedups_and_resolves_conflicts():
    store = _store()
    store.add_user_memory("u1", "lives in Beijing", metadata={"tags": ["home"]})
    store.add_user_memory("u1", "likes window seats", metadata={"tags": ["seat"]})
    store.add_user_memory(
        "u1", "likes window seats", metadata={"tags": ["seat"]}
    )  # dup
    store.add_user_memory(
        "u1", "lives in Shanghai", metadata={"tags": ["home"]}
    )  # conflict

    report = service.consolidate_memories(store, "u1")
    assert report["initial"] == 4
    assert report["duplicates_removed"] == 1
    assert report["conflicts_resolved"][0]["attribute"] == "home"
    assert report["conflicts_resolved"][0]["kept"] == "lives in Shanghai"
    assert report["final"] == 2
    texts = {r.memory_text for r in store.get_user_memory_records("u1")}
    assert texts == {"lives in Shanghai", "likes window seats"}
    store.close()
