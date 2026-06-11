"""Unit tests for LocalFilePollConnector, LocalFileSlimConnector, and
LocalFileSlimConnectorWithPermSync."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.internal.connectors import (
    LocalFileConnector,
    LocalFilePollConnector,
    LocalFileSlimConnector,
    LocalFileSlimConnectorWithPermSync,
    SlimDocument,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str = "content") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _docs(connector) -> list:
    return [doc for batch in connector.load_from_state() for doc in batch]


def _slim_docs(connector, **kwargs) -> list[SlimDocument]:
    return [
        doc for batch in connector.retrieve_all_slim_docs(**kwargs) for doc in batch
    ]


def _slim_perm_docs(connector, **kwargs) -> list[SlimDocument]:
    return [
        doc
        for batch in connector.retrieve_all_slim_docs_perm_sync(**kwargs)
        for doc in batch
    ]


# ---------------------------------------------------------------------------
# LocalFilePollConnector
# ---------------------------------------------------------------------------


def test_poll_connector_is_also_load_connector(tmp_path):
    _write(tmp_path / "a.txt")
    connector = LocalFilePollConnector([tmp_path])
    # Inherits load_from_state from LocalFileConnector
    assert isinstance(connector, LocalFileConnector)
    docs = _docs(connector)
    assert [d.title for d in docs] == ["a.txt"]


def test_poll_connector_yields_only_files_in_window(tmp_path):
    old = _write(tmp_path / "old.txt", "old")
    new = _write(tmp_path / "new.txt", "new")

    now = time.time()
    # Back-date old.txt so its mtime is clearly before the window
    import os

    os.utime(old, (now - 200, now - 200))
    os.utime(new, (now - 10, now - 10))

    start = now - 60
    end = now

    connector = LocalFilePollConnector([tmp_path])
    results = [doc for batch in connector.poll_source(start, end) for doc in batch]

    assert [r.title for r in results] == ["new.txt"]


def test_poll_connector_empty_window_yields_nothing(tmp_path):
    _write(tmp_path / "a.txt")
    connector = LocalFilePollConnector([tmp_path])
    results = list(connector.poll_source(start=0.0, end=1.0))
    assert results == []


def test_poll_connector_respects_allowed_extensions(tmp_path):
    _write(tmp_path / "keep.txt")
    _write(tmp_path / "skip.log")

    now = time.time()
    start, end = now - 60, now + 60

    connector = LocalFilePollConnector([tmp_path], allowed_extensions=[".txt"])
    results = [doc for batch in connector.poll_source(start, end) for doc in batch]

    assert [r.title for r in results] == ["keep.txt"]


# ---------------------------------------------------------------------------
# LocalFileSlimConnector
# ---------------------------------------------------------------------------


def test_slim_connector_yields_slim_documents(tmp_path):
    _write(tmp_path / "a.txt")
    _write(tmp_path / "b.md")

    slim = LocalFileSlimConnector([tmp_path])
    docs = _slim_docs(slim)

    assert len(docs) == 2
    assert all(isinstance(d, SlimDocument) for d in docs)
    assert all("path" in d.metadata for d in docs)
    assert all("mtime" in d.metadata for d in docs)
    assert all("size" in d.metadata for d in docs)


def test_slim_connector_does_not_read_file_content(tmp_path, monkeypatch):
    _write(tmp_path / "a.txt", "should not be read")

    read_calls: list[str] = []
    original_read_text = Path.read_text

    def tracking_read_text(self, *args, **kwargs):
        read_calls.append(str(self))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracking_read_text)

    slim = LocalFileSlimConnector([tmp_path])
    _slim_docs(slim)

    assert read_calls == [], "SlimConnector must not read file contents"


def test_slim_connector_skips_unsupported_extensions(tmp_path):
    _write(tmp_path / "keep.txt")
    _write(tmp_path / "skip.bin")

    docs = _slim_docs(LocalFileSlimConnector([tmp_path]))
    assert len(docs) == 1
    assert docs[0].metadata["path"].endswith("keep.txt")


def test_full_and_slim_connectors_deduplicate_overlapping_paths(tmp_path):
    file = _write(tmp_path / "a.txt")

    full_docs = _docs(LocalFileConnector([tmp_path, file]))
    slim_docs = _slim_docs(LocalFileSlimConnector([tmp_path, file]))

    assert [doc.id for doc in full_docs] == [str(file.resolve())]
    assert [doc.id for doc in slim_docs] == [str(file.resolve())]


def test_slim_connector_requires_paths():
    with pytest.raises(ValueError, match="at least one path"):
        list(LocalFileSlimConnector().retrieve_all_slim_docs())


def test_slim_connector_time_window_filtering(tmp_path):
    import os

    old = _write(tmp_path / "old.txt")
    new = _write(tmp_path / "new.txt")

    now = time.time()
    os.utime(old, (now - 200, now - 200))
    os.utime(new, (now - 5, now - 5))

    docs = _slim_docs(LocalFileSlimConnector([tmp_path]), start=now - 60, end=now)

    assert len(docs) == 1
    assert docs[0].metadata["path"].endswith("new.txt")


def test_slim_connector_no_permissions_in_base_class(tmp_path):
    _write(tmp_path / "a.txt")
    docs = _slim_docs(LocalFileSlimConnector([tmp_path]))
    assert docs[0].permissions == {}


def test_slim_connector_batch_size(tmp_path):
    for i in range(5):
        _write(tmp_path / f"{i}.txt")

    batches = list(
        LocalFileSlimConnector([tmp_path], batch_size=2).retrieve_all_slim_docs()
    )
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 5


# ---------------------------------------------------------------------------
# LocalFileSlimConnectorWithPermSync
# ---------------------------------------------------------------------------


def test_perm_sync_connector_includes_permissions(tmp_path):
    _write(tmp_path / "a.txt")

    docs = _slim_perm_docs(LocalFileSlimConnectorWithPermSync([tmp_path]))

    assert len(docs) == 1
    perms = docs[0].permissions
    assert "readable" in perms
    assert "writable" in perms
    assert "executable" in perms
    assert "mode" in perms
    assert isinstance(perms["readable"], bool)
    assert perms["mode"].startswith("0o")


def test_perm_sync_readable_file_is_marked_readable(tmp_path):
    _write(tmp_path / "r.txt")
    docs = _slim_perm_docs(LocalFileSlimConnectorWithPermSync([tmp_path]))
    assert docs[0].permissions["readable"] is True


def test_perm_sync_connector_inherits_time_window_filtering(tmp_path):
    import os

    old = _write(tmp_path / "old.txt")
    new = _write(tmp_path / "new.txt")

    now = time.time()
    os.utime(old, (now - 200, now - 200))
    os.utime(new, (now - 5, now - 5))

    docs = _slim_perm_docs(
        LocalFileSlimConnectorWithPermSync([tmp_path]), start=now - 60, end=now
    )

    assert len(docs) == 1
    assert docs[0].metadata["path"].endswith("new.txt")
    assert "readable" in docs[0].permissions


def test_perm_sync_connector_also_has_slim_retrieve(tmp_path):
    _write(tmp_path / "a.txt")
    connector = LocalFileSlimConnectorWithPermSync([tmp_path])
    # retrieve_all_slim_docs (no perms) still works
    docs = _slim_docs(connector)
    assert len(docs) == 1
    assert docs[0].permissions == {}
