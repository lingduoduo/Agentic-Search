# MCP Document Extraction Design

## Goal

Convert the sampled document-processing code into a native Agentic Search MCP
tool. The implementation will follow the repository's FastMCP registration,
plain-dictionary response, optional-dependency, documentation, and testing
patterns rather than preserving the sampled standalone perception-server API.

## Scope

The MCP server will expose one authenticated tool named `extract_document`.
It will extract content from PDF, DOCX, PPTX, CSV, and plain-text documents
supplied directly in the MCP request.

The initial implementation will not accept server-local paths or remote URLs.
Allowing either would let an MCP caller reach files or network locations
outside the document supplied for processing. Existing indexed-document search
and retrieval tools remain unchanged.

## MCP Interface

`extract_document` accepts:

- `file_name: str`: the original filename, used to select and report the format;
- `content_base64: str`: base64-encoded document bytes;
- `page_range: str | None`: an optional one-based PDF page selection such as
  `1-3,5`;
- `max_rows: int`: the maximum number of CSV data rows to return.

The tool returns a normal dictionary, consistent with existing MCP tools:

```text
{
  "document": {
    "file_name": str,
    "file_type": str,
    "text": str,
    ... format-specific fields
  }
}
```

Failures return:

```text
{
  "error": str,
  "document": null
}
```

Error text will be actionable but will not expose tracebacks, temporary paths,
or other host details.

## Architecture

The MCP adapter will live in
`src/internal/mcp_server/tools/documents.py` and register
`extract_document` with the repository's shared `mcp_server` instance.
`src/internal/mcp_server/api.py` and the tools package will import the module so
the decorator executes during server startup.

Format-specific extraction will be implemented as undecorated helper
functions. The MCP function will:

1. validate the filename, format-specific options, and encoded-input size;
2. decode the payload with strict base64 validation;
3. dispatch by a case-insensitive filename extension;
4. invoke the appropriate parser;
5. normalize and bound the result;
6. return the repository-style dictionary payload.

The sampled `base.py` and `document_processing.py` are reference inputs, not
the final module boundaries. Their useful parsing behavior will be adapted,
while `ActionResponse`, `TextContent`, URL downloading, unrestricted local-path
validation, `load_dotenv`, emoji logging, and traceback serialization will not
be carried into the MCP implementation.

## Format Behavior

- PDF: return selected page text, total page count, and extracted page count.
  Page selections are one-based, deduplicated, sorted, and bounded by the
  document. Malformed or descending ranges are rejected.
- DOCX: return nonblank paragraph text and table cell data.
- PPTX: return text grouped by slide and report both total and nonblank slides.
- CSV: return column names and JSON-compatible records up to `max_rows`.
  Truncation is determined by reading one extra row, avoiding the sampled
  implementation's false positive when a file has exactly `max_rows` rows.
- TXT: decode UTF-8 text. Invalid UTF-8 returns an error instead of silently
  replacing bytes.

All text fields will use one shared output-character limit. Format-specific
nested data will also be bounded so the overall MCP response cannot bypass the
limit through tables, slides, or CSV records.

## Safety and Resource Limits

The tool will reject:

- unsupported or missing extensions;
- invalid base64;
- empty input;
- encoded or decoded input above the configured fixed limit;
- invalid page ranges and nonpositive or excessive CSV row limits.

Parsers that require filesystem input will use a temporary file with the
validated suffix. Cleanup will occur in a `finally` block. Parser work is
synchronous and potentially CPU-heavy, so the async MCP entry point will run it
off the event loop through `asyncio.to_thread`.

No URL fetcher or arbitrary path reader will be exposed.

## Dependencies

Parser libraries will be declared in a dedicated optional project extra for
MCP document extraction. Importing the MCP server must continue to work when
those libraries are absent. Selecting a format whose parser is unavailable
will return an installation-oriented error naming the required extra.

TXT extraction has no additional parser dependency.

## Testing

Implementation will follow test-driven development. Focused unit tests will
first demonstrate failures for:

- MCP registration and schema visibility;
- TXT extraction and strict base64/UTF-8 validation;
- format dispatch and unsupported extensions;
- page-range parsing, including malformed, descending, duplicate, and
  out-of-bounds selections;
- CSV exact-limit and over-limit truncation;
- output and input limits;
- missing optional dependencies;
- temporary-file cleanup after success and parser failure;
- consistent, non-sensitive error payloads.

PDF, DOCX, and PPTX behavior will use small generated fixtures when their
optional libraries are installed, with dependency-boundary tests covering the
unavailable case. Validation will include the focused unit suite, existing MCP
unit tests, default repository tests where practical, FastMCP tool discovery,
and a direct MCP tool invocation.

## Documentation

The MCP README will list `extract_document`, its supported formats, arguments,
limits, optional installation extra, example request, response shape, and the
explicit absence of local-path and URL support.
