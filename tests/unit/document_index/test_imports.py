"""Verify every document_index file can be imported without error."""


def test_document_index_utils_importable():
    import src.internal.document_index.document_index_utils  # noqa: F401
