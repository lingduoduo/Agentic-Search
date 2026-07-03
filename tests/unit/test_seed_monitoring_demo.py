from __future__ import annotations

import json

from src.internal.db import AgenticSearchStore
from src.internal.db.models import ConnectorConfig


def test_seed_creates_expected_database_and_matching_corpus(tmp_path):
    from examples.seed_monitoring_demo import seed_monitoring_demo

    db_path = tmp_path / "nested" / "demo.sqlite3"
    corpus_path = tmp_path / "output" / "corpus.jsonl"

    summary = seed_monitoring_demo(db_path, corpus_path)

    assert summary == {
        "db_path": str(db_path.resolve()),
        "corpus_path": str(corpus_path.resolve()),
        "enabled_connectors": 2,
        "documents": 6,
        "attempts": 4,
    }
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    assert len(rows) == 6

    store = AgenticSearchStore(db_path)
    documents = {document.id: document for document in store.list_documents()}
    store.close()
    assert {row["id"]: (row["title"], row["contents"], row["url"]) for row in rows} == {
        document.id: (document.title, document.contents, document.url)
        for document in documents.values()
    }


def test_seed_is_idempotent_and_preserves_unrelated_rows(tmp_path):
    from examples.seed_monitoring_demo import seed_monitoring_demo

    db_path = tmp_path / "demo.sqlite3"
    corpus_path = tmp_path / "corpus.jsonl"
    store = AgenticSearchStore(db_path)
    store.upsert_connector(
        ConnectorConfig(id="existing", name="Existing", source="test")
    )
    store.close()

    first = seed_monitoring_demo(db_path, corpus_path)
    second = seed_monitoring_demo(db_path, corpus_path)

    assert first == second
    store = AgenticSearchStore(db_path)
    assert store.get_connector("existing") is not None
    assert len(store.list_index_attempts()) == 4
    store.close()


def test_main_rejects_in_memory_database(monkeypatch, capsys):
    from examples.seed_monitoring_demo import main

    monkeypatch.delenv("AGENTIC_SEARCH_WEB_DB_PATH", raising=False)

    assert main([]) == 2
    assert "--db-path" in capsys.readouterr().err


def test_main_uses_configured_database_and_prints_json(monkeypatch, tmp_path, capsys):
    from examples.seed_monitoring_demo import main

    db_path = tmp_path / "configured.sqlite3"
    corpus_path = tmp_path / "configured.jsonl"
    monkeypatch.setenv("AGENTIC_SEARCH_WEB_DB_PATH", str(db_path))

    assert main(["--corpus-path", str(corpus_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["db_path"] == str(db_path.resolve())
    assert summary["corpus_path"] == str(corpus_path.resolve())
