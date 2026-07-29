"""Tests for MCP document extraction input and TXT handling."""

import base64

import pytest

from src.internal.mcp_server.tools import documents


@pytest.mark.asyncio
async def test_extract_document_decodes_txt():
    """TXT content is decoded and returned with its metadata."""
    result = await documents.extract_document(
        "notes.TXT", base64.b64encode(b"alpha\nbeta").decode()
    )

    assert result == {
        "document": {
            "file_name": "notes.TXT",
            "file_type": "txt",
            "text": "alpha\nbeta",
            "text_length": 10,
            "truncated": False,
        }
    }


@pytest.mark.parametrize(
    ("file_name", "payload", "error_fragment"),
    [
        ("", "YQ==", "file name"),
        ("no-extension", "YQ==", "extension"),
        ("data.bin", "YQ==", "unsupported"),
        ("data.txt", "%%%", "base64"),
        ("data.txt", "", "empty"),
    ],
)
@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_input(
    file_name: str, payload: str, error_fragment: str
):
    """Malformed requests receive bounded error responses."""
    result = await documents.extract_document(file_name, payload)

    assert result["document"] is None
    assert error_fragment in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_utf8_txt():
    """TXT input must be UTF-8 rather than silently replacing bad bytes."""
    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"\xff").decode()
    )

    assert result["document"] is None
    assert "utf-8" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_input_larger_than_limit(monkeypatch):
    """Decoded input larger than the configured limit is rejected."""
    monkeypatch.setattr(documents, "MAX_INPUT_BYTES", 3)

    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"four").decode()
    )

    assert result["document"] is None
    assert "input limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_truncates_text_larger_than_output_limit(monkeypatch):
    """TXT responses expose truncation while retaining the original length."""
    monkeypatch.setattr(documents, "MAX_OUTPUT_CHARS", 3)

    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"four").decode()
    )

    assert result == {
        "document": {
            "file_name": "notes.txt",
            "file_type": "txt",
            "text": "fou",
            "text_length": 4,
            "truncated": True,
        }
    }
