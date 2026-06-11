"""Verify every document_index file can be imported without error."""


def test_disabled_importable():
    import src.internal.document_index.disabled  # noqa: F401


def test_interfaces_importable():
    import src.internal.document_index.interfaces  # noqa: F401


def test_document_metadata_importable():
    import src.internal.document_index.document_metadata  # noqa: F401


def test_document_index_utils_importable():
    import src.internal.document_index.document_index_utils  # noqa: F401


def test_chunk_content_enrichment_importable():
    import src.internal.document_index.chunk_content_enrichment  # noqa: F401


def test_opensearch_constants_importable():
    import src.internal.document_index.opensearch.constants  # noqa: F401


def test_opensearch_schema_importable():
    import src.internal.document_index.opensearch.schema  # noqa: F401


def test_opensearch_search_importable():
    import src.internal.document_index.opensearch.search  # noqa: F401


def test_opensearch_client_importable():
    import src.internal.document_index.opensearch.client  # noqa: F401


def test_opensearch_document_index_importable():
    import src.internal.document_index.opensearch.opensearch_document_index  # noqa: F401
