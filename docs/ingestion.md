# Ingestion

[← Back to README](../README.md)

This guide covers how documents get into the system before query time.
For the repository layout and the end-to-end request flow, see
[Architecture](architecture.md); for chunking, embedding, and index internals,
see [Retrieval](retrieval.md).

> **Note:** the async connector/background-worker ingestion pipeline previously
> documented here has been removed from this repo. What remains is the
> connector data-model + management API surface, and the offline `index_builder`
> tool described below.

## Connectors

`src/internal/connectors/models.py` defines the shared document models
(`Document`, `SlimDocument`, etc.) used by connector integrations; the
connector *implementations* that produced them — and the connector/document
management endpoints and their ingestion tables — have since been removed.
OAuth authorization is still managed through the `oauth/` router registered in
`create_web_app()` (see the [HTTP API reference](api-reference.md)).

## Building search indexes

Searchable indexes are built offline with the `index_builder` CLI, documented in
[Retrieval](retrieval.md#retrieval-setup):

```bash
python3 -m src.internal.document_index.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir data/indexes/
```

The retrieval servers (`src/internal/servers/retrieval/`) then read the
indexes it produces. The search and chat paths only read these indexes; they
never index on the request path.
