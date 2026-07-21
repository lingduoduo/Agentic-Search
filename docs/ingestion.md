# Ingestion

[← Back to README](../README.md)

This guide covers how documents get into the system before query time:
connectors, document ingestion, indexing, and the background workers that run it.
For the repository layout and the end-to-end request flow, see
[Architecture](architecture.md); for chunking, embedding, and index internals,
see [Retrieval](retrieval.md).

## Pipeline at a glance

```text
async connectors / ingestion jobs
  → chunk + embed/index documents
  → searchable retrieval indexes
```

Searchable documents are prepared **before** query time by asynchronous ingestion
and indexing jobs. The search and chat paths read the indexes these jobs produce;
they never index on the request path.

## Connectors

Connectors (`src/internal/connectors/`) pull documents from data sources into the
pipeline. Connector setup, credentials, and OAuth authorization are managed
through the admin/enterprise routers registered in `create_web_app()`
(`connectors/`, `documents/`, `oauth/`, and `indexing/` endpoints — see the
[HTTP API reference](api-reference.md)).

## Background processing

Ingestion and indexing run as asynchronous workers under
`src/internal/servers/backgroundworker/`:

- **`beat_worker`** — cron-style scheduling of recurring jobs.
- **`light_worker`** — polling and job scheduling.
- **`docfetching` / `docprocessing`** — fetch source documents and normalize them.
- **`heavy_worker`** — chunking, embedding, and index building.
- **`monitoring_worker`** — health and progress monitoring.
- **`user_file_processing_worker`** — ingestion of user-uploaded files.

The low-level chunking, embedding, and FAISS index-building these workers invoke
live in `src/internal/document_index/` and `src/internal/retrieval/`, and are
documented in [Retrieval](retrieval.md).
