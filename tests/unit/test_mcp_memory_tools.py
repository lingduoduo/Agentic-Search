from types import SimpleNamespace

import pytest


def test_memory_module_imports_and_registers_tools():
    # Importing the tools module must register the tools on the shared server
    from src.internal.mcp_server import api  # noqa: F401
    from src.internal.mcp_server.tools import memory  # noqa: F401

    assert hasattr(memory, "save_memory")
    assert hasattr(memory, "update_memory_from_conversation")
    assert hasattr(memory, "search_memories")


def test_resolve_user_id_defaults_without_token(monkeypatch):
    from src.internal.mcp_server.tools import memory
    from src.internal.memory.service import DEFAULT_MEMORY_USER_ID

    def _raise():
        raise ValueError("no token")

    monkeypatch.setattr(memory, "require_access_token", _raise)
    assert memory._resolve_user_id() == DEFAULT_MEMORY_USER_ID


def test_old_stub_removed():
    import importlib.util

    assert importlib.util.find_spec("src.internal.mcp_server.memory") is None


def test_resolve_user_id_refuses_anonymous_under_strict_mode(monkeypatch):
    """MCP is a second door to the same bucket; strict mode must close it too.

    Enforcing only in the web router would leave the flag half-honoured: an
    operator who set it would still have an unauthenticated MCP client writing
    into the shared ``default_user`` memories.
    """
    from src.internal.mcp_server.tools import memory

    def _raise():
        raise ValueError("no token")

    monkeypatch.setattr(memory, "require_access_token", _raise)
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH", "1")
    with pytest.raises(PermissionError):
        memory._resolve_user_id()


def test_strict_mode_still_honours_a_token(monkeypatch):
    from src.internal.mcp_server.tools import memory

    monkeypatch.setattr(
        memory,
        "require_access_token",
        lambda: SimpleNamespace(claims={"sub": "alice"}),
    )
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH", "1")
    assert memory._resolve_user_id() == "alice"
