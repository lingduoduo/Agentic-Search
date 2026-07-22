from src.internal.memory import service


def test_maybe_build_encoder_off_by_default(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", raising=False)
    assert service.maybe_build_encoder() is None


def test_maybe_build_encoder_none_when_disabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", "no")
    assert service.maybe_build_encoder() is None
