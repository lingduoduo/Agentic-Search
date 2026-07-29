"""Document extraction tool for the Agentic Search MCP server."""

from __future__ import annotations

import asyncio
import base64
import binascii
from pathlib import PurePath
from typing import Any

from ..api import mcp_server

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_CHARS = 50_000
MAX_CSV_ROWS = 10_000
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}


def _error(message: str) -> dict[str, Any]:
    return {"error": message, "document": None}


def _decode_document(content_base64: str) -> bytes:
    if not content_base64:
        raise ValueError("Document content must not be empty.")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Document content is not valid base64.") from exc
    if not data:
        raise ValueError("Decoded document must not be empty.")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("Decoded document exceeds the 20 MiB input limit.")
    return data


def _bounded_text(text: str) -> tuple[str, int, bool]:
    return text[:MAX_OUTPUT_CHARS], len(text), len(text) > MAX_OUTPUT_CHARS


def _extract_document_sync(
    file_name: str, data: bytes, page_range: str | None, max_rows: int
) -> dict[str, Any]:
    if not file_name or PurePath(file_name).name != file_name:
        raise ValueError("Document file name must be a simple file name.")

    extension = PurePath(file_name).suffix.lower()
    if not extension:
        raise ValueError("Document file name must include an extension.")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {extension}.")
    if extension != ".txt":
        raise ValueError(f"Document format is not available yet: {extension}.")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("TXT document content must be valid UTF-8.") from exc

    bounded_text, text_length, truncated = _bounded_text(text)
    return {
        "document": {
            "file_name": file_name,
            "file_type": "txt",
            "text": bounded_text,
            "text_length": text_length,
            "truncated": truncated,
        }
    }


@mcp_server.tool()
async def extract_document(
    file_name: str,
    content_base64: str,
    page_range: str | None = None,
    max_rows: int = 1000,
) -> dict[str, Any]:
    """Extract bounded text content from an uploaded document."""
    try:
        data = _decode_document(content_base64)
        return await asyncio.to_thread(
            _extract_document_sync, file_name, data, page_range, max_rows
        )
    except (ValueError, ImportError) as exc:
        return _error(str(exc))
    except Exception:
        return _error("Document extraction failed.")
