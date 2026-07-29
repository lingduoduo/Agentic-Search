"""Bounded document extraction for the Agentic Search MCP server."""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import importlib
import io
import json
import multiprocessing
import sys
import threading
import time
import zipfile
from pathlib import PurePath
from typing import Any, Callable
from urllib.parse import urlsplit

from .. import document_parser_runtime
from ..api import mcp_server

# Request/response bounds. The encoded cap is the largest padded base64 string
# that can represent MAX_INPUT_BYTES and is checked before decoding allocates.
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_ENCODED_INPUT_CHARS = 4 * ((MAX_INPUT_BYTES + 2) // 3)
MAX_RESPONSE_CHARS = 50_000
MAX_ERROR_CHARS = 256
MAX_FILE_NAME_CHARS = 255
MAX_PAGE_RANGE_CHARS = 256
MAX_CSV_ROWS = 10_000

# Office Open XML is ZIP-based. These preflight limits allow normal documents
# while rejecting archive bombs before python-docx/python-pptx see the bytes.
MAX_OFFICE_ZIP_ENTRIES = 5_000
MAX_OFFICE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_OFFICE_COMPRESSION_RATIO = 100

# Untrusted PDF/Office parsers run in disposable child processes. Linux applies
# hard kernel limits; macOS/Windows use the parent watchdog below.
MAX_PARSER_PROCESSES = 2
PARSER_TIMEOUT_SECONDS = 15.0
PARSER_CPU_SECONDS = 10
PARSER_MEMORY_BYTES = 1024 * 1024 * 1024
PARSER_TERMINATE_GRACE_SECONDS = 1.0
PARSER_WATCHDOG_INTERVAL_SECONDS = 0.05

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}
_PARSER_START_METHOD = (
    "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
)
_PARSER_CONTEXT = multiprocessing.get_context(_PARSER_START_METHOD)
_PARSER_SLOTS = threading.BoundedSemaphore(MAX_PARSER_PROCESSES)
_USES_PARENT_RESOURCE_WATCHDOG = not sys.platform.startswith("linux")


class DocumentExtractionError(Exception):
    """A bounded, caller-safe document extraction failure."""


ParserDependencyError = document_parser_runtime.ParserDependencyError


class ParserTimeoutError(DocumentExtractionError):
    """A parser exceeded its wall-clock deadline."""


class ParserResourceError(DocumentExtractionError):
    """A parser exceeded or could not enforce a resource boundary."""


class ParserExecutionError(DocumentExtractionError):
    """A parser failed without exposing child-process details."""


def _serialized_length(value: Any) -> int:
    """Return the conservative JSON character cost of an MCP response."""
    return len(json.dumps(value, ensure_ascii=False))


def _error(message: str) -> dict[str, Any]:
    """Build an error payload whose message and complete envelope are bounded."""
    safe_message = str(message)[:MAX_ERROR_CHARS] or "Document extraction failed."
    payload: dict[str, Any] = {"error": safe_message, "document": None}
    if _serialized_length(payload) <= MAX_RESPONSE_CHARS:
        return payload

    payload["error"] = "Document extraction failed."
    if _serialized_length(payload) <= MAX_RESPONSE_CHARS:
        return payload
    # The production response cap is far above this invariant floor. Retaining
    # the normal response schema is preferable to returning an unbounded value.
    return {"error": "", "document": None}


def _finalize_response(result: dict[str, Any]) -> dict[str, Any]:
    """Centrally enforce the advertised cap on every success/error envelope."""
    if _serialized_length(result) <= MAX_RESPONSE_CHARS:
        return result
    return _error("Document response exceeds the output limit.")


def _success_fits(document: dict[str, Any]) -> bool:
    return _serialized_length({"document": document}) <= MAX_RESPONSE_CHARS


def _decode_document(content_base64: str) -> bytes:
    if not isinstance(content_base64, str) or not content_base64:
        raise ValueError("Document content must not be empty.")
    if len(content_base64) > MAX_ENCODED_INPUT_CHARS:
        raise ValueError("Document content exceeds the encoded input limit.")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Document content is not valid base64.") from exc
    if not data:
        raise ValueError("Decoded document must not be empty.")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("Decoded document exceeds the 20 MiB input limit.")
    return data


def _validate_file_name(file_name: str) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Document file name must be a simple file name.")
    if len(file_name) > MAX_FILE_NAME_CHARS:
        raise ValueError("Document file name exceeds the length limit.")
    if (
        PurePath(file_name).name != file_name
        or "\\" in file_name
        or urlsplit(file_name).scheme
    ):
        raise ValueError("Document file name must be a simple file name.")

    extension = PurePath(file_name).suffix.lower()
    if not extension:
        raise ValueError("Document file name must include an extension.")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported document format.")
    return extension


def _validate_page_range_option(page_range: str | None, extension: str) -> None:
    if page_range is None:
        return
    if extension != ".pdf":
        raise ValueError("Page range is only supported for PDF documents.")
    if not isinstance(page_range, str) or not page_range:
        raise ValueError("PDF page range must not be empty.")
    if len(page_range) > MAX_PAGE_RANGE_CHARS:
        raise ValueError("PDF page range exceeds the length limit.")

    for segment in page_range.split(","):
        bounds = segment.split("-")
        if len(bounds) not in (1, 2) or not all(
            bound.isdecimal() and int(bound) > 0 for bound in bounds
        ):
            raise ValueError("Invalid PDF page range.")
        if int(bounds[0]) > int(bounds[-1]):
            raise ValueError("Invalid PDF page range.")


def parse_page_range(page_range: str, total_pages: int) -> list[int]:
    """Return zero-based, sorted PDF indexes selected by a one-based range."""
    if total_pages < 1:
        raise ValueError("PDF document must contain at least one page.")
    _validate_page_range_option(page_range, ".pdf")

    pages: set[int] = set()
    for segment in page_range.split(","):
        bounds = segment.split("-")
        start = int(bounds[0])
        end = int(bounds[-1])
        pages.update(range(start - 1, min(end, total_pages)))
    return sorted(pages)


def _fit_text_document(document: dict[str, Any], text: str) -> dict[str, Any]:
    """Fit a text field inside its complete success response envelope."""
    document["text"] = ""
    document["text_length"] = len(text)
    document["truncated"] = False
    if not _success_fits(document):
        raise ValueError("Document metadata exceeds the output limit.")

    low, high = 0, len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        document["text"] = text[:midpoint]
        document["truncated"] = midpoint < len(text)
        if _success_fits(document):
            low = midpoint
        else:
            high = midpoint - 1

    document["text"] = text[:low]
    document["truncated"] = low < len(text)
    return {"document": document}


_extract_pdf = document_parser_runtime.extract_pdf
_extract_docx = document_parser_runtime.extract_docx
_extract_pptx = document_parser_runtime.extract_pptx


def _extract_csv(data: bytes, file_name: str, max_rows: int) -> dict[str, Any]:
    """Extract CSV headings and rows within the complete response budget."""
    if not isinstance(max_rows, int) or isinstance(max_rows, bool):
        raise ValueError("CSV row limit must be an integer.")
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
    document: dict[str, Any] = {
        "file_name": file_name,
        "file_type": "csv",
        "columns": reader.fieldnames,
        "records": [],
        "rows": 0,
        "truncated": len(rows) > max_rows,
    }
    if not _success_fits(document):
        raise ValueError("CSV column headings exceed the output limit.")

    for row in rows[:max_rows]:
        document["records"].append(row)
        document["rows"] = len(document["records"])
        if not _success_fits(document):
            document["records"].pop()
            document["rows"] = len(document["records"])
            document["truncated"] = True
            break
    return {"document": document}


def _preflight_office_zip(data: bytes) -> None:
    """Reject oversized or highly compressed Office ZIPs before parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("Office document is not a valid ZIP archive.") from exc

    if len(entries) > MAX_OFFICE_ZIP_ENTRIES:
        raise ValueError("Office document ZIP contains too many entries.")

    uncompressed_bytes = sum(entry.file_size for entry in entries)
    compressed_bytes = sum(entry.compress_size for entry in entries)
    if uncompressed_bytes > MAX_OFFICE_UNCOMPRESSED_BYTES:
        raise ValueError("Office document exceeds the expanded content limit.")
    if any(entry.flag_bits & 0x1 for entry in entries):
        raise ValueError("Encrypted Office ZIP entries are unsupported.")

    for entry in entries:
        if entry.file_size and (
            entry.compress_size == 0
            or entry.file_size / entry.compress_size > MAX_OFFICE_COMPRESSION_RATIO
        ):
            raise ValueError("Office document exceeds the compression ratio limit.")
    if uncompressed_bytes and (
        compressed_bytes == 0
        or uncompressed_bytes / compressed_bytes > MAX_OFFICE_COMPRESSION_RATIO
    ):
        raise ValueError("Office document exceeds the compression ratio limit.")


def _terminate_process(process: multiprocessing.Process) -> None:
    """Terminate, escalate to kill when available, and always reap the child."""
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(PARSER_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
        else:  # pragma: no cover - modern supported Python exposes kill()
            process.terminate()
        process.join()


def _require_watchdog_module() -> Any:
    """Load the RSS/CPU watchdog dependency or fail closed."""
    try:
        return importlib.import_module("psutil")
    except ImportError as exc:
        raise ParserResourceError(
            "Document parser resource limits are unavailable."
        ) from exc


def _watchdog_limit_exceeded(psutil: Any, process_id: int) -> bool:
    """Sample aggregate parser RSS and CPU use without exposing host details."""
    try:
        root = psutil.Process(process_id)
        processes = [root, *root.children(recursive=True)]
        rss_bytes = sum(item.memory_info().rss for item in processes)
        cpu_seconds = sum(
            item.cpu_times().user + item.cpu_times().system for item in processes
        )
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as exc:
        raise ParserResourceError(
            "Document parser resource limits are unavailable."
        ) from exc
    return rss_bytes > PARSER_MEMORY_BYTES or cpu_seconds > PARSER_CPU_SECONDS


def _run_parser_in_process(
    parser: Callable[..., dict[str, Any]], *args: Any
) -> dict[str, Any]:
    """Run one parser behind concurrency, wall-time, and resource boundaries."""
    if not _PARSER_SLOTS.acquire(timeout=PARSER_TIMEOUT_SECONDS):
        raise ParserTimeoutError("Document parser timed out.")

    receive_connection = send_connection = None
    process = None
    try:
        watchdog = (
            _require_watchdog_module() if _USES_PARENT_RESOURCE_WATCHDOG else None
        )
        receive_connection, send_connection = _PARSER_CONTEXT.Pipe(duplex=False)
        process = _PARSER_CONTEXT.Process(
            target=document_parser_runtime.parser_worker,
            args=(
                send_connection,
                parser,
                args,
                PARSER_CPU_SECONDS,
                PARSER_MEMORY_BYTES,
                not _USES_PARENT_RESOURCE_WATCHDOG,
            ),
            daemon=True,
        )
        process.start()
        send_connection.close()
        send_connection = None

        deadline = time.monotonic() + PARSER_TIMEOUT_SECONDS
        message_ready = False
        while process.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                raise ParserTimeoutError("Document parser timed out.")
            if receive_connection.poll(
                min(PARSER_WATCHDOG_INTERVAL_SECONDS, remaining)
            ):
                message_ready = True
                break
            if watchdog is not None and _watchdog_limit_exceeded(watchdog, process.pid):
                _terminate_process(process)
                raise ParserResourceError(
                    "Document parser exceeded its resource limit."
                )
        if not message_ready:
            message_ready = receive_connection.poll(PARSER_WATCHDOG_INTERVAL_SECONDS)
        if not message_ready:
            _terminate_process(process)
            raise ParserResourceError("Document parser exceeded its resource limit.")

        try:
            status, payload = receive_connection.recv()
        except EOFError as exc:
            _terminate_process(process)
            raise ParserResourceError(
                "Document parser exceeded its resource limit."
            ) from exc

        process.join(PARSER_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            _terminate_process(process)
            raise ParserTimeoutError("Document parser timed out.")
        if status == "ok":
            return payload
        if status == "dependency":
            raise ParserDependencyError(payload)
        if status == "resource":
            raise ParserResourceError("Document parser exceeded its resource limit.")
        if status == "unavailable":
            raise ParserResourceError(
                "Document parser resource limits are unavailable."
            )
        raise ParserExecutionError("Document parser failed.")
    finally:
        if process is not None and process.is_alive():
            _terminate_process(process)
        if send_connection is not None:
            send_connection.close()
        if receive_connection is not None:
            receive_connection.close()
        _PARSER_SLOTS.release()


def _extract_document_sync(
    file_name: str, data: bytes, page_range: str | None, max_rows: int
) -> dict[str, Any]:
    extension = _validate_file_name(file_name)
    _validate_page_range_option(page_range, extension)

    if extension == ".pdf":
        return _run_parser_in_process(
            document_parser_runtime.extract_pdf,
            data,
            file_name,
            page_range,
            MAX_RESPONSE_CHARS,
        )
    if extension in {".docx", ".pptx"}:
        _preflight_office_zip(data)
        parser = (
            document_parser_runtime.extract_docx
            if extension == ".docx"
            else document_parser_runtime.extract_pptx
        )
        return _run_parser_in_process(parser, data, file_name, MAX_RESPONSE_CHARS)
    if extension == ".csv":
        return _extract_csv(data, file_name, max_rows)

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("TXT document content must be valid UTF-8.") from exc
    return _fit_text_document(
        {"file_name": file_name, "file_type": "txt"},
        text,
    )


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
        result = await asyncio.to_thread(
            _extract_document_sync, file_name, data, page_range, max_rows
        )
        return _finalize_response(result)
    except (ValueError, ImportError, DocumentExtractionError) as exc:
        return _finalize_response(_error(str(exc)))
    except Exception:
        return _finalize_response(_error("Document extraction failed."))
