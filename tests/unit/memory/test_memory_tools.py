import asyncio

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.tools import build_memory_registry


def test_memory_tools_mutate_store_and_count():
    store = AgenticSearchStore(":memory:")
    registry, counts, schemas = build_memory_registry(store, "u1")
    assert {s["function"]["name"] for s in schemas} == {
        "add_memory",
        "update_memory",
        "delete_memory",
    }

    async def run():
        resp, _raw, errs = await registry.invoke("add_memory", {"content": "likes tea"})
        assert not errs
        mem_id = store.get_user_memory_records("u1")[0].id
        await registry.invoke(
            "update_memory", {"memory_id": mem_id, "content": "likes green tea"}
        )
        await registry.invoke("delete_memory", {"memory_id": mem_id})

    asyncio.run(run())
    assert counts == {"add": 1, "update": 1, "delete": 1}
    assert store.get_user_memory_records("u1") == []
    store.close()


def test_add_memory_schema_validation_rejects_missing_content():
    store = AgenticSearchStore(":memory:")
    registry, counts, _ = build_memory_registry(store, "u1")

    async def run():
        _resp, _raw, errs = await registry.invoke("add_memory", {})
        return errs

    errs = asyncio.run(run())
    assert errs  # schema requires "content"
    assert counts["add"] == 0
    store.close()
