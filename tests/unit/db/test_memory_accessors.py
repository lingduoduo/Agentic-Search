# tests/unit/db/test_memory_accessors.py
from src.internal.db.store import AgenticSearchStore


def test_update_and_delete_memory_by_id():
    store = AgenticSearchStore(":memory:")
    a = store.add_user_memory("u1", "likes window seats")
    b = store.add_user_memory("u1", "lives in Beijing")

    # get_user_memory_records returns records with ids, active only
    ids = {r.id for r in store.get_user_memory_records("u1")}
    assert ids == {a.id, b.id}

    # update by id
    updated = store.update_user_memory("u1", a.id, "prefers aisle seats")
    assert updated is not None and updated.memory_text == "prefers aisle seats"

    # wrong user cannot update
    assert store.update_user_memory("other", a.id, "hacked") is None

    # delete by id is a soft delete (drops from active list)
    assert store.delete_user_memory("u1", b.id) is True
    remaining = [r.memory_text for r in store.get_user_memory_records("u1")]
    assert remaining == ["prefers aisle seats"]

    # deleting again returns False (already inactive)
    assert store.delete_user_memory("u1", b.id) is False
    store.close()
