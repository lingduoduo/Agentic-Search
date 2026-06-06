"""Verify every document_index file can be imported without error."""


def test_vespa_constants_importable():
    import src.backend.document_index.vespa_constants  # noqa: F401


def test_disabled_importable():
    import src.backend.document_index.disabled  # noqa: F401


def test_interfaces_new_importable():
    import src.backend.document_index.interfaces_new  # noqa: F401


def test_document_metadata_importable():
    import src.backend.document_index.document_metadata  # noqa: F401


def test_document_index_utils_importable():
    import src.backend.document_index.document_index_utils  # noqa: F401


def test_chunk_content_enrichment_importable():
    import src.backend.document_index.chunk_content_enrichment  # noqa: F401
