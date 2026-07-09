"""Guard: dead AgentState memory fields stay removed."""

from __future__ import annotations

from src.agents.core.state import AgentState


def test_dead_memory_fields_removed():
    fields = AgentState.__dataclass_fields__
    assert "short_term_memory" not in fields
    assert "long_term_memory" not in fields
