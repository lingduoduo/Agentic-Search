def test_factory_importable():
    import src.internal.document_index.factory  # noqa: F401


def test_get_default_index_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    import src.internal.document_index.factory as factory_mod
    from src.internal.document_index.disabled import DisabledDocumentIndex

    idx = factory_mod.get_default_document_index(
        primary_index_name="test_index",
        secondary_index_name=None,
    )
    assert isinstance(idx, DisabledDocumentIndex)


def test_disabled_verify_is_noop(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    import src.internal.document_index.factory as factory_mod

    idx = factory_mod.get_default_document_index(
        primary_index_name="test_index",
        secondary_index_name=None,
    )
    # verify_and_create_index_if_necessary should be a no-op
    from src.internal.document_index.models import EmbeddingPrecision

    idx.verify_and_create_index_if_necessary(
        embedding_dim=768, embedding_precision=EmbeddingPrecision.FLOAT
    )


def test_get_all_indices_disabled_returns_list(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    import src.internal.document_index.factory as factory_mod
    from src.internal.document_index.disabled import DisabledDocumentIndex

    indices = factory_mod.get_all_document_indices(
        primary_index_name="test_index",
        secondary_index_name=None,
    )
    assert len(indices) == 1
    assert isinstance(indices[0], DisabledDocumentIndex)


def test_factory_no_db_session_param():
    """Factory functions must not require a db_session parameter."""
    import inspect
    import src.internal.document_index.factory as factory_mod

    for func_name in ("get_default_document_index", "get_all_document_indices"):
        func = getattr(factory_mod, func_name)
        sig = inspect.signature(func)
        assert "db_session" not in sig.parameters, f"{func_name} has db_session"
        assert "session" not in sig.parameters, f"{func_name} has session"


def test_is_opensearch_enabled_reads_new_env_var(monkeypatch):
    """Factory must respond to ENABLE_OPENSEARCH_INDEXING."""
    import src.internal.document_index.factory as factory_mod

    monkeypatch.setenv("ENABLE_OPENSEARCH_INDEXING", "true")
    assert factory_mod._is_opensearch_enabled() is True
