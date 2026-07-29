"""Tests for MCP document extraction input and TXT handling."""

import base64
import io

import pytest

from src.internal.mcp_server.tools import documents


def _docx_payload() -> str:
    """Create a small DOCX payload with paragraph and table content."""
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("paragraph text")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "left"
    table.cell(0, 1).text = "right"
    output = io.BytesIO()
    document.save(output)
    return base64.b64encode(output.getvalue()).decode()


def _pptx_payload(slide_texts: tuple[str, ...] = ("slide text", "")) -> str:
    """Create a small PPTX payload with optional text on each slide."""
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    for text in slide_texts:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        if text:
            slide.shapes.add_textbox(0, 0, 100, 100).text = text
    output = io.BytesIO()
    presentation.save(output)
    return base64.b64encode(output.getvalue()).decode()


@pytest.mark.parametrize(
    ("raw", "total", "expected"),
    [
        ("1", 5, [0]),
        ("1-3,5", 5, [0, 1, 2, 4]),
        ("3,1,3", 5, [0, 2]),
        ("1-99", 3, [0, 1, 2]),
        ("99", 3, []),
    ],
)
def test_parse_page_range(raw: str, total: int, expected: list[int]):
    """PDF page ranges are one-based, sorted, deduplicated, and bounded."""
    assert documents.parse_page_range(raw, total) == expected


@pytest.mark.parametrize("page_range", ["", "0", "-1", "3-1", "1-", "1--2", "a"])
def test_parse_page_range_rejects_malformed_ranges(page_range: str):
    """Malformed PDF page ranges are rejected before extraction."""
    with pytest.raises(ValueError):
        documents.parse_page_range(page_range, 3)


def test_parse_page_range_rejects_empty_documents():
    """Page selection is invalid for a PDF with no pages."""
    with pytest.raises(ValueError):
        documents.parse_page_range("1", 0)


def _two_page_pdf() -> bytes:
    """Build a small text PDF without a non-PDF test dependency."""
    pypdf2 = pytest.importorskip("PyPDF2")
    generic = pypdf2.generic
    writer = pypdf2.PdfWriter()

    for text in ("alpha", "beta"):
        writer.add_blank_page(width=200, height=200)
        page = writer.pages[-1]
        resources = generic.DictionaryObject()
        font = generic.DictionaryObject(
            {
                generic.NameObject("/Type"): generic.NameObject("/Font"),
                generic.NameObject("/Subtype"): generic.NameObject("/Type1"),
                generic.NameObject("/BaseFont"): generic.NameObject("/Helvetica"),
            }
        )
        resources[generic.NameObject("/Font")] = generic.DictionaryObject(
            {generic.NameObject("/F1"): writer._add_object(font)}
        )
        page[generic.NameObject("/Resources")] = resources
        stream = generic.DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[generic.NameObject("/Contents")] = writer._add_object(stream)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_pdf_returns_selected_labeled_pages():
    """PDF extraction returns bounded labeled text and page-count metadata."""
    result = documents._extract_pdf(_two_page_pdf(), "notes.pdf", "2")

    assert result == {
        "document": {
            "file_name": "notes.pdf",
            "file_type": "pdf",
            "text": "--- Page 2 ---\nbeta",
            "text_length": 19,
            "truncated": False,
            "total_pages": 2,
            "extracted_pages": 1,
        }
    }


def test_extract_pdf_reports_missing_optional_dependency(monkeypatch):
    """Absent PDF support explains how to install the document extra."""

    def missing_module(module_name: str):
        raise ImportError(module_name)

    monkeypatch.setattr(documents.importlib, "import_module", missing_module)

    with pytest.raises(ImportError, match=r"agentic-search\[mcp-documents\]"):
        documents._require_module("PyPDF2", "PyPDF2")


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


@pytest.mark.asyncio
async def test_extract_document_rejects_page_ranges_for_non_pdf_documents():
    """Page ranges apply only to PDF documents."""
    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"alpha").decode(), page_range="1"
    )

    assert result["document"] is None
    assert "page range" in result["error"].lower()


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


@pytest.mark.parametrize(
    "file_name",
    [
        "C:\\uploads\\notes.txt",
        "\\\\server\\share\\notes.txt",
        "https://example.com/notes.txt",
    ],
)
@pytest.mark.asyncio
async def test_extract_document_rejects_path_or_uri_file_names(file_name: str):
    """Document names must be simple basenames on every client platform."""
    result = await documents.extract_document(
        file_name, base64.b64encode(b"alpha").decode()
    )

    assert result["document"] is None
    assert "file name" in result["error"].lower()


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


@pytest.mark.asyncio
async def test_extract_document_extracts_docx_paragraphs_and_tables():
    """DOCX output includes nonblank paragraphs and JSON-compatible table cells."""
    result = await documents.extract_document("notes.docx", _docx_payload())

    assert result["document"] == {
        "file_name": "notes.docx",
        "file_type": "docx",
        "paragraphs": ["paragraph text"],
        "tables": [[["left", "right"]]],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_extract_document_extracts_pptx_slide_text():
    """PPTX output groups shape text by slide and counts nonblank slides."""
    result = await documents.extract_document("slides.pptx", _pptx_payload())

    assert result["document"] == {
        "file_name": "slides.pptx",
        "file_type": "pptx",
        "slides": [{"slide": 1, "text": "slide text"}],
        "total_slides": 2,
        "nonblank_slides": 1,
        "returned_slides": 1,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_csv_exact_row_limit_is_not_truncated():
    """A CSV at the requested row limit is complete rather than truncated."""
    payload = base64.b64encode(b"name,value\na,1\nb,2\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is False


@pytest.mark.asyncio
async def test_csv_over_row_limit_is_truncated():
    """A CSV with an extra record reports truncation after the requested limit."""
    payload = base64.b64encode(b"name,value\na,1\nb,2\nc,3\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is True


@pytest.mark.asyncio
async def test_extract_document_extracts_csv_columns_and_records():
    """CSV records retain headings and values as JSON-compatible dictionaries."""
    payload = base64.b64encode(b"name,value\na,1\n").decode()
    result = await documents.extract_document("rows.csv", payload)

    assert result["document"] == {
        "file_name": "rows.csv",
        "file_type": "csv",
        "columns": ["name", "value"],
        "records": [{"name": "a", "value": "1"}],
        "rows": 1,
        "truncated": False,
    }


@pytest.mark.parametrize("max_rows", [0, -1, documents.MAX_CSV_ROWS + 1])
@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_csv_row_limits(max_rows: int):
    """CSV row limits must be positive and within the configured maximum."""
    payload = base64.b64encode(b"name\na\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=max_rows)

    assert result["document"] is None
    assert "row limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_utf8_csv():
    """CSV input must be valid UTF-8 rather than silently replacing bytes."""
    result = await documents.extract_document(
        "rows.csv", base64.b64encode(b"name\n\xff\n").decode()
    )

    assert result["document"] is None
    assert "utf-8" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_duplicate_csv_headings():
    """Duplicate CSV headings cannot safely become dictionary keys."""
    payload = base64.b64encode(b"name,name\na,b\n").decode()
    result = await documents.extract_document("rows.csv", payload)

    assert result["document"] is None
    assert "duplicate" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_bounds_csv_records(monkeypatch):
    """CSV records stop before nested output can exceed the shared character budget."""
    monkeypatch.setattr(documents, "MAX_OUTPUT_CHARS", 20)
    payload = base64.b64encode(b"name\nfirst\nsecond\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert result["document"]["records"] == [{"name": "first"}]
    assert result["document"]["rows"] == 1
    assert result["document"]["truncated"] is True


@pytest.mark.asyncio
async def test_extract_document_bounds_docx_tables(monkeypatch):
    """DOCX table cells stop accumulating after the shared character budget."""
    monkeypatch.setattr(documents, "MAX_OUTPUT_CHARS", 20)
    result = await documents.extract_document("notes.docx", _docx_payload())

    assert result["document"]["tables"] == []
    assert result["document"]["truncated"] is True


@pytest.mark.asyncio
async def test_extract_document_bounds_pptx_slides(monkeypatch):
    """PPTX slide objects stop accumulating after the shared character budget."""
    monkeypatch.setattr(documents, "MAX_OUTPUT_CHARS", 20)
    result = await documents.extract_document("slides.pptx", _pptx_payload())

    assert result["document"]["slides"] == []
    assert result["document"]["truncated"] is True


@pytest.mark.asyncio
async def test_extract_document_reports_complete_pptx_slide_counts_after_truncation(
    monkeypatch,
):
    """PPTX metadata distinguishes all nonblank slides from returned slides."""
    monkeypatch.setattr(documents, "MAX_OUTPUT_CHARS", 30)
    result = await documents.extract_document(
        "slides.pptx", _pptx_payload(("a", "b", ""))
    )

    assert result["document"]["total_slides"] == 3
    assert result["document"]["nonblank_slides"] == 2
    assert result["document"]["returned_slides"] == 1
    assert result["document"]["truncated"] is True


@pytest.mark.parametrize(
    ("extension", "module_name", "distribution_name"),
    [
        ("docx", "docx", "python-docx"),
        ("pptx", "pptx", "python-pptx"),
    ],
)
@pytest.mark.asyncio
async def test_extract_document_reports_missing_office_dependency(
    monkeypatch, extension: str, module_name: str, distribution_name: str
):
    """Missing optional office parsers return actionable installation errors."""

    def missing_module(name: str):
        if name == module_name:
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(documents.importlib, "import_module", missing_module)
    result = await documents.extract_document(
        f"notes.{extension}", base64.b64encode(b"x").decode()
    )

    assert result["document"] is None
    assert distribution_name in result["error"]
