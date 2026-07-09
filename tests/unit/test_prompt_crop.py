"""Unit tests for system-preserving prompt-id truncation."""

from __future__ import annotations

from src.agents.core.base import AgentLoopBase, AgentLoopConfig, _crop_prompt_ids
from tests.unit.test_agent_loop import DummyServerManager, DummyTokenizerWithEncode


def test_under_budget_unchanged():
    full = [1, 2, 3]
    assert _crop_prompt_ids(full, [9], 10) == full
    assert _crop_prompt_ids(full, [9], 0) == full  # budget <= 0 → unchanged


def test_no_system_tail_crop():
    full = list(range(10))
    assert _crop_prompt_ids(full, [], 4) == [6, 7, 8, 9]


def test_system_preserved_over_budget():
    system = [100, 101]
    full = list(range(20))  # far over budget
    out = _crop_prompt_ids(full, system, 6)
    assert len(out) == 6
    assert out[:2] == system
    assert out[2:] == full[-(6 - 2) :]  # recent tail fills the rest


def test_system_larger_than_budget_degenerate():
    system = [1, 2, 3, 4, 5]
    full = list(range(50))
    out = _crop_prompt_ids(full, system, 3)
    assert out == system[-3:]


def test_build_prompt_ids_sync_keeps_system_prefix():
    loop = AgentLoopBase(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=40),
    )
    system_content = "SYSTEM RULES"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "u" * 200},  # forces the crop
    ]
    ids = loop._build_prompt_ids_sync(messages)
    assert len(ids) == 40
    assert ids[: len(system_content)] == [ord(c) for c in system_content]
