"""Lightweight child-process runtime for untrusted document parsers.

This module intentionally does not import the MCP server. Spawned children can
therefore apply meaningful memory limits without first loading FastMCP and the
rest of the application.
"""

from __future__ import annotations

import importlib
import io
import json
from typing import Any, Callable


class ParserDependencyError(ImportError):
    """A selected optional parser is not installed."""


class ResourceLimitsUnavailable(RuntimeError):
    """The host refused the required child resource limits."""


def _serialized_length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def _success_fits(document: dict[str, Any], response_cap: int) -> bool:
    return _serialized_length({"document": document}) <= response_cap


def _require_module(module_name: str, distribution_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ParserDependencyError(
            f"{distribution_name} is required; install agentic-search[mcp-documents]."
        ) from exc


def parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """Return bounded zero-based PDF indexes after parent-side syntax checks."""
    if total_pages < 1:
        raise ValueError("PDF document must contain at least one page.")
    pages: set[int] = set()
    for segment in page_range.split(","):
        bounds = segment.split("-")
        start = int(bounds[0])
        end = int(bounds[-1])
        pages.update(range(start - 1, min(end, total_pages)))
    return sorted(pages)


def _fit_text_document(
    document: dict[str, Any], text: str, response_cap: int
) -> dict[str, Any]:
    document["text"] = ""
    document["text_length"] = len(text)
    document["truncated"] = False
    if not _success_fits(document, response_cap):
        raise ValueError("Document metadata exceeds the output limit.")

    low, high = 0, len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        document["text"] = text[:midpoint]
        document["truncated"] = midpoint < len(text)
        if _success_fits(document, response_cap):
            low = midpoint
        else:
            high = midpoint - 1
    document["text"] = text[:low]
    document["truncated"] = low < len(text)
    return {"document": document}


def extract_pdf(
    data: bytes,
    file_name: str,
    page_range: str | None,
    response_cap: int = 50_000,
) -> dict[str, Any]:
    """Extract selected PDF pages using the maintained optional pypdf package."""
    pypdf = _require_module("pypdf", "pypdf")
    reader = pypdf.PdfReader(io.BytesIO(data))
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
    return _fit_text_document(
        {
            "file_name": file_name,
            "file_type": "pdf",
            "total_pages": total_pages,
            "extracted_pages": len(page_indexes),
        },
        text,
        response_cap,
    )


def extract_docx(
    data: bytes, file_name: str, response_cap: int = 50_000
) -> dict[str, Any]:
    """Extract nonblank paragraphs and table rows within the response cap."""
    docx = _require_module("docx", "python-docx")
    parsed = docx.Document(io.BytesIO(data))
    document: dict[str, Any] = {
        "file_name": file_name,
        "file_type": "docx",
        "paragraphs": [],
        "tables": [],
        "truncated": False,
    }
    if not _success_fits(document, response_cap):
        raise ValueError("Document metadata exceeds the output limit.")

    for paragraph in parsed.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        document["paragraphs"].append(text)
        if not _success_fits(document, response_cap):
            document["paragraphs"].pop()
            document["truncated"] = True
            break

    if not document["truncated"]:
        for table in parsed.tables:
            extracted_table: list[list[str]] = []
            for row in table.rows:
                extracted_table.append([cell.text.strip() for cell in row.cells])
                document["tables"].append(extracted_table)
                fits = _success_fits(document, response_cap)
                document["tables"].pop()
                if not fits:
                    extracted_table.pop()
                    document["truncated"] = True
                    break
            if extracted_table:
                document["tables"].append(extracted_table)
            if document["truncated"]:
                break
    return {"document": document}


def extract_pptx(
    data: bytes, file_name: str, response_cap: int = 50_000
) -> dict[str, Any]:
    """Extract nonblank slide text within the response cap."""
    pptx = _require_module("pptx", "python-pptx")
    presentation = pptx.Presentation(io.BytesIO(data))
    slide_texts: list[tuple[int, str]] = []
    for number, slide in enumerate(presentation.slides, start=1):
        text = "\n".join(
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        )
        if text:
            slide_texts.append((number, text))

    document: dict[str, Any] = {
        "file_name": file_name,
        "file_type": "pptx",
        "slides": [],
        "total_slides": len(presentation.slides),
        "nonblank_slides": len(slide_texts),
        "returned_slides": 0,
        "truncated": False,
    }
    if not _success_fits(document, response_cap):
        raise ValueError("Document metadata exceeds the output limit.")

    for number, text in slide_texts:
        document["slides"].append({"slide": number, "text": text})
        document["returned_slides"] = len(document["slides"])
        if not _success_fits(document, response_cap):
            document["slides"].pop()
            document["returned_slides"] = len(document["slides"])
            document["truncated"] = True
            break
    return {"document": document}


def _apply_resource_limits(
    cpu_seconds: int, memory_bytes: int, apply_hard_memory_limit: bool
) -> None:
    """Apply kernel limits where supported; the parent watches other hosts."""
    try:
        import resource
    except ImportError:
        if apply_hard_memory_limit:
            raise ResourceLimitsUnavailable("resource limits unavailable")
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if apply_hard_memory_limit:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (OSError, ValueError) as exc:
        raise ResourceLimitsUnavailable("resource limits unavailable") from exc


def parser_worker(
    connection: Any,
    parser: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    cpu_seconds: int,
    memory_bytes: int,
    apply_hard_memory_limit: bool,
) -> None:
    """Execute one parser and return only bounded status across the pipe."""
    try:
        _apply_resource_limits(cpu_seconds, memory_bytes, apply_hard_memory_limit)
        connection.send(("ok", parser(*args)))
    except ParserDependencyError as exc:
        connection.send(("dependency", str(exc)))
    except ResourceLimitsUnavailable:
        connection.send(("unavailable", None))
    except MemoryError:
        connection.send(("resource", None))
    except BaseException:
        connection.send(("error", None))
    finally:
        connection.close()
