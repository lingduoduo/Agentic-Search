from src.context.models import SearchContextBundle
from src.context.prompts import build_answer_prompt, build_structured_answer_prompt
from src.internal.db.store import AgenticSearchStore
from src.internal.memory.service import MEMORY_INJECTION_MAX, memory_preamble


def test_memory_preamble_empty_user_returns_blank():
    store = AgenticSearchStore(":memory:")
    assert memory_preamble(store, "nobody") == ""
    store.close()


def test_memory_preamble_formats_instructional_block():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User prefers window seats")
    pre = memory_preamble(store, "u1")
    assert pre.startswith("\n\nWhat you know about this user")
    assert "allergic to peanuts" in pre
    assert "- User prefers window seats" in pre
    store.close()


def test_memory_preamble_caps_to_most_recent():
    store = AgenticSearchStore(":memory:")
    for i in range(MEMORY_INJECTION_MAX + 5):
        store.add_user_memory("u1", f"memory number {i}")
    pre = memory_preamble(store, "u1", max_items=MEMORY_INJECTION_MAX)
    # exactly MEMORY_INJECTION_MAX bullet lines, and the most-recent ones kept
    assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    assert f"memory number {MEMORY_INJECTION_MAX + 4}" in pre  # newest
    assert "memory number 0" not in pre  # oldest dropped
    store.close()


def test_prompt_builders_inject_memory_into_system():
    ctx = SearchContextBundle(query="q", documents=[])
    pre = "\n\nWhat you know about this user:\n- User is allergic to peanuts"

    answer = build_answer_prompt("Recommend Thai food", ctx, user_memory=pre)
    assert "allergic to peanuts" in answer.system

    structured = build_structured_answer_prompt(
        "Recommend Thai food", ctx, user_memory=pre
    )
    assert "allergic to peanuts" in structured.system


def test_prompt_builders_unchanged_without_memory():
    ctx = SearchContextBundle(query="q", documents=[])
    assert "allergic" not in build_answer_prompt("q", ctx).system
    assert "allergic" not in build_structured_answer_prompt("q", ctx).system
