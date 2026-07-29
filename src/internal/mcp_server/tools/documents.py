"""Document extraction tool for the Agentic Search MCP server."""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import importlib
import io
import json
from pathlib import PurePath
from typing import Any
from urllib.parse import urlsplit

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


def _serialized_length(value: Any) -> int:
    """Return the JSON character cost of a value in an MCP response."""
    return len(json.dumps(value, ensure_ascii=False))


def _require_module(module_name: str, distribution_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{distribution_name} is required; install agentic-search[mcp-documents]."
        ) from exc


def parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """Return zero-based, sorted PDF page indexes selected by a one-based range."""
    if total_pages < 1:
        raise ValueError("PDF document must contain at least one page.")
    if not page_range:
        raise ValueError("PDF page range must not be empty.")

    pages: set[int] = set()
    for segment in page_range.split(","):
        bounds = segment.split("-")
        if len(bounds) not in (1, 2) or not all(
            bound.isdecimal() and int(bound) > 0 for bound in bounds
        ):
            raise ValueError(f"Invalid PDF page range: {page_range}.")

        start = int(bounds[0])
        end = int(bounds[-1])
        if start > end:
            raise ValueError(f"Invalid PDF page range: {page_range}.")
        pages.update(range(start - 1, min(end, total_pages)))

    return sorted(pages)


def _extract_pdf(data: bytes, file_name: str, page_range: str | None) -> dict[str, Any]:
    """Extract selected PDF pages using the optional PyPDF2 dependency."""
    pypdf2 = _require_module("PyPDF2", "PyPDF2")
    reader = pypdf2.PdfReader(io.BytesIO(data))
    total_pages = len(reader.pages)
    page_indexes = (
        list(range(total_pages))
        if page_range is None
        else parse_page_range(page_range, total_pages)
    )
    text = "\n\n".join(
        f"--- Page {page_index + 1} ---\n{reader.pages[page_index].extract_text() or ''}"
        for page_index in page_indexes
    )
    bounded_text, text_length, truncated = _bounded_text(text)
    return {
        "document": {
            "file_name": file_name,
            "file_type": "pdf",
            "text": bounded_text,
            "text_length": text_length,
            "truncated": truncated,
            "total_pages": total_pages,
            "extracted_pages": len(page_indexes),
        }
    }


def _extract_docx(data: bytes, file_name: str) -> dict[str, Any]:
    """Extract nonblank paragraphs and bounded table rows from a DOCX file."""
    docx = _require_module("docx", "python-docx")
    document = docx.Document(io.BytesIO(data))
    paragraphs: list[str] = []
    tables: list[list[list[str]]] = []
    used_chars = 0
    truncated = False

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        cost = _serialized_length(text)
        if used_chars + cost > MAX_OUTPUT_CHARS:
            truncated = True
            break
        paragraphs.append(text)
        used_chars += cost

    for table in document.tables:
        extracted_table: list[list[str]] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            cost = _serialized_length(cells)
            if used_chars + cost > MAX_OUTPUT_CHARS:
                truncated = True
                break
            extracted_table.append(cells)
            used_chars += cost
        if extracted_table:
            tables.append(extracted_table)
        if truncated:
            break

    return {
        "document": {
            "file_name": file_name,
            "file_type": "docx",
            "paragraphs": paragraphs,
            "tables": tables,
            "truncated": truncated,
        }
    }


def _extract_pptx(data: bytes, file_name: str) -> dict[str, Any]:
    """Extract bounded, nonblank text grouped by slide from a PPTX file."""
    pptx = _require_module("pptx", "python-pptx")
    presentation = pptx.Presentation(io.BytesIO(data))
    slides: list[dict[str, Any]] = []
    used_chars = 0
    truncated = False
    nonblank_slides = 0

    for number, slide in enumerate(presentation.slides, start=1):
        text = "\n".join(
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        )
        if not text:
            continue
        nonblank_slides += 1
        extracted_slide = {"slide": number, "text": text}
        cost = _serialized_length(extracted_slide)
        if used_chars + cost > MAX_OUTPUT_CHARS:
            truncated = True
            break
        slides.append(extracted_slide)
        used_chars += cost

    return {
        "document": {
            "file_name": file_name,
            "file_type": "pptx",
            "slides": slides,
            "total_slides": len(presentation.slides),
            "extracted_slides": nonblank_slides,
            "truncated": truncated,
        }
    }


def _extract_csv(data: bytes, file_name: str, max_rows: int) -> dict[str, Any]:
    """Extract a bounded number of CSV records with validated headings."""
    if not 1 <= max_rows <= MAX_CSV_ROWS:
        raise ValueError(f"CSV row limit must be between 1 and {MAX_CSV_ROWS}.")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV document content must be valid UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or any(not heading for heading in reader.fieldnames):
        raise ValueError("CSV document must include column headings.")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV document must not contain duplicate column headings.")

    rows = [row for _, row in zip(range(max_rows + 1), reader, strict=False)]
    records: list[dict[str, str | None]] = []
    used_chars = 0
    truncated = len(rows) > max_rows
    for row in rows[:max_rows]:
        cost = _serialized_length(row)
        if used_chars + cost > MAX_OUTPUT_CHARS:
            truncated = True
            break
        records.append(row)
        used_chars += cost

    return {
        "document": {
            "file_name": file_name,
            "file_type": "csv",
            "columns": reader.fieldnames,
            "records": records,
            "rows": len(records),
            "truncated": truncated,
        }
    }


def _extract_document_sync(
    file_name: str, data: bytes, page_range: str | None, max_rows: int
) -> dict[str, Any]:
    if (
        not file_name
        or PurePath(file_name).name != file_name
        or "\\" in file_name
        or urlsplit(file_name).scheme
    ):
        raise ValueError("Document file name must be a simple file name.")

    extension = PurePath(file_name).suffix.lower()
    if not extension:
        raise ValueError("Document file name must include an extension.")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {extension}.")
    if extension != ".pdf" and page_range is not None:
        raise ValueError("Page range is only supported for PDF documents.")
    if extension == ".pdf":
        return _extract_pdf(data, file_name, page_range)
    if extension == ".docx":
        return _extract_docx(data, file_name)
    if extension == ".pptx":
        return _extract_pptx(data, file_name)
    if extension == ".csv":
        return _extract_csv(data, file_name, max_rows)

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
