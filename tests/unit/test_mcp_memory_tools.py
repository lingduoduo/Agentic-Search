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
