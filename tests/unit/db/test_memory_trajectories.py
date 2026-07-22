# tests/unit/db/test_memory_trajectories.py
from src.internal.db.store import AgenticSearchStore


def test_trajectory_round_trip_newest_first():
    store = AgenticSearchStore(":memory:")
    store.add_memory_trajectory(
        "u1",
        session_id="s1",
        model="m",
        trajectory={
            "memory_before": [],
            "tool_calls": [{"name": "add_memory"}],
            "memory_after": ["x"],
            "counts": {"add": 1},
        },
    )
    store.add_memory_trajectory(
        "u1", session_id=None, model="m", trajectory={"counts": {}}
    )
    got = store.list_memory_trajectories("u1")
    assert len(got) == 2
    assert got[0].session_id is None  # newest first
    assert got[1].trajectory["counts"] == {"add": 1}
    store.close()
