"""Populate a file-backed SQLite store for the monitoring dashboard demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.internal.db import AgenticSearchStore
from src.internal.db.models import ConnectorConfig, StoredDocument

DEFAULT_CORPUS_PATH = Path("data/monitoring_demo_corpus.jsonl")

CONNECTORS = (
    ConnectorConfig(
        id="monitoring-demo-docs",
        name="Product Docs",
        source="filesystem",
        enabled=True,
        metadata={"demo": True},
    ),
    ConnectorConfig(
        id="monitoring-demo-support",
        name="Support Knowledge",
        source="web",
        enabled=True,
        metadata={"demo": True},
    ),
    ConnectorConfig(
        id="monitoring-demo-archive",
        name="Archive",
        source="filesystem",
        enabled=False,
        metadata={"demo": True},
    ),
)

DOCUMENTS = (
    StoredDocument(
        id="monitoring-demo-architecture",
        title="Agentic Search Architecture",
        contents="The system separates retrieval, web orchestration, and the React dashboard.",
        url="https://demo.local/architecture",
        connector_id="monitoring-demo-docs",
        metadata={"demo": True, "collection": "Product Docs"},
    ),
    StoredDocument(
        id="monitoring-demo-monitoring",
        title="Monitoring Guide",
        contents="The Console shows server health, worker metrics, query transforms, and retrieval comparisons.",
        url="https://demo.local/monitoring",
        connector_id="monitoring-demo-docs",
        metadata={"demo": True, "collection": "Product Docs"},
    ),
    StoredDocument(
        id="monitoring-demo-retrieval",
        title="Retrieval Operations",
        contents="Sparse retrieval uses TF-IDF in the local demo server and returns ranked source documents.",
        url="https://demo.local/retrieval",
        connector_id="monitoring-demo-docs",
        metadata={"demo": True, "collection": "Product Docs"},
    ),
    StoredDocument(
        id="monitoring-demo-indexing",
        title="Indexing Runbook",
        contents="Index attempts progress from not_started to in_progress and then success or failed.",
        url="https://demo.local/indexing",
        connector_id="monitoring-demo-support",
        metadata={"demo": True, "collection": "Support Knowledge"},
    ),
    StoredDocument(
        id="monitoring-demo-grounding",
        title="Grounding Troubleshooting",
        contents="Citations indicate retrieval grounding while answer text indicates synthesis success.",
        url="https://demo.local/grounding",
        connector_id="monitoring-demo-support",
        metadata={"demo": True, "collection": "Support Knowledge"},
    ),
    StoredDocument(
        id="monitoring-demo-workers",
        title="Worker Capacity",
        contents="Queue depth and active connector counts provide a live operational snapshot.",
        url="https://demo.local/workers",
        connector_id="monitoring-demo-support",
        metadata={"demo": True, "collection": "Support Knowledge"},
    ),
)

ATTEMPTS = (
    ("monitoring-demo-pending", "monitoring-demo-docs", "not_started", 0, 0, None),
    ("monitoring-demo-running", "monitoring-demo-support", "in_progress", 3, 12, None),
    ("monitoring-demo-success", "monitoring-demo-docs", "success", 6, 24, None),
    (
        "monitoring-demo-failed",
        "monitoring-demo-archive",
        "failed",
        1,
        2,
        "Source temporarily unavailable",
    ),
)


def seed_monitoring_demo(
    db_path: str | Path, corpus_path: str | Path
) -> dict[str, object]:
    """Seed demo records and write their matching retrieval corpus."""
    if str(db_path) == ":memory:":
        raise ValueError("monitoring demo requires a file-backed SQLite database")

    resolved_db = Path(db_path).expanduser().resolve()
    resolved_corpus = Path(corpus_path).expanduser().resolve()
    resolved_db.parent.mkdir(parents=True, exist_ok=True)
    resolved_corpus.parent.mkdir(parents=True, exist_ok=True)

    store = AgenticSearchStore(resolved_db)
    try:
        for connector in CONNECTORS:
            store.upsert_connector(connector)
        for document in DOCUMENTS:
            store.upsert_document(document)
        for attempt in ATTEMPTS:
            attempt_id, connector_id, status, document_count, chunk_count, error = (
                attempt
            )
            if store.get_index_attempt(attempt_id) is None:
                store.create_index_attempt(
                    attempt_id=attempt_id,
                    connector_id=connector_id,
                    status=status,
                    total_documents=document_count,
                    total_chunks=chunk_count,
                    error=error,
                    metadata={"demo": True},
                )

        summary = {
            "db_path": str(resolved_db),
            "corpus_path": str(resolved_corpus),
            "enabled_connectors": len(store.list_connectors(enabled=True)),
            "documents": len(store.list_documents()),
            "attempts": len(store.list_index_attempts()),
        }
    finally:
        store.close()

    corpus_rows = (
        {
            "id": document.id,
            "title": document.title,
            "contents": document.contents,
            "url": document.url,
        }
        for document in DOCUMENTS
    )
    temporary_corpus = resolved_corpus.with_suffix(resolved_corpus.suffix + ".tmp")
    temporary_corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus_rows)
    )
    temporary_corpus.replace(resolved_corpus)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    args = parser.parse_args(argv)

    db_path = args.db_path or os.getenv("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:")
    if db_path == ":memory:":
        print(
            "A file-backed SQLite database is required; pass --db-path or set "
            "AGENTIC_SEARCH_WEB_DB_PATH.",
            file=sys.stderr,
        )
        return 2

    summary = seed_monitoring_demo(db_path, args.corpus_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
