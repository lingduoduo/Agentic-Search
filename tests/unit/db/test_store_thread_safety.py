"""`AgenticSearchStore` must survive concurrent access from multiple threads.

The store holds ONE `sqlite3` connection opened with `check_same_thread=False`
and shares it across every caller. That is reachable in production with two
users: `_finalize_response` writes from the event-loop thread while
`POST /api/sessions` and `GET /api/sessions/{id}` are plain `def` handlers, which
FastAPI runs in its anyio worker threadpool.

Without serialization that combination does not merely slow down -- it raises.
Measured against the unguarded store:

    sqlite3.InterfaceError: bad parameter or other API misuse
    sqlite3.DatabaseError: cannot start a transaction within a transaction
    IndexError: tuple index out of range        (row parsed as a bare tuple)

Per-statement locking would not be enough: the store issues 39 `commit()` calls,
so a second thread can interleave a commit between two statements of another
thread's transaction. The lock has to span the whole public method.
"""

from __future__ import annotations

import threading

import pytest

from src.internal.db import AgenticSearchStore

_THREADS = 8
_TURNS = 25


@pytest.fixture()
def store(tmp_path):
    s = AgenticSearchStore(str(tmp_path / "concurrent.db"))
    yield s
    s.close()


def _drive(store, session_id: str, errors: list[BaseException]) -> None:
    """One session's worth of write-then-read traffic, as a request would do."""
    try:
        for turn in range(_TURNS):
            store.add_chat_message(session_id, role="user", content=f"q{turn}" * 40)
            store.add_chat_message(
                session_id, role="assistant", content=f"a{turn}" * 40
            )
            store.list_chat_messages(session_id)
    except BaseException as exc:  # noqa: BLE001 - the assertion is "nothing raised"
        errors.append(exc)


def test_concurrent_sessions_do_not_corrupt_the_shared_connection(store):
    """Eight concurrent sessions, no exceptions, and every write lands."""
    session_ids = [store.create_chat_session(title=f"s{i}").id for i in range(_THREADS)]
    errors: list[BaseException] = []

    threads = [
        threading.Thread(target=_drive, args=(store, sid, errors))
        for sid in session_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, (
        f"{len(errors)} thread(s) raised against the shared connection; "
        f"first: {type(errors[0]).__name__}: {errors[0]}"
    )
    # Not just "no exception": every write has to be durable and attributed to
    # the right session. A lock that dropped writes would still pass the above.
    for sid in session_ids:
        assert len(store.list_chat_messages(sid)) == _TURNS * 2


def test_concurrent_readers_and_writers_across_sessions(store):
    """Readers on one session while writers hammer others.

    Closer to the real interleaving than uniform traffic: the request path mixes
    `GET /api/sessions/{id}` reads with stream-finalize writes.
    """
    writer_ids = [store.create_chat_session(title=f"w{i}").id for i in range(4)]
    reader_id = store.create_chat_session(title="reader").id
    for turn in range(10):
        store.add_chat_message(reader_id, role="user", content=f"seed{turn}")

    errors: list[BaseException] = []

    def read_loop() -> None:
        try:
            for _ in range(100):
                assert len(store.list_chat_messages(reader_id)) == 10
                store.get_chat_session(reader_id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=read_loop) for _ in range(4)]
    threads += [
        threading.Thread(target=_drive, args=(store, sid, errors)) for sid in writer_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, (
        f"{len(errors)} thread(s) raised; first: "
        f"{type(errors[0]).__name__}: {errors[0]}"
    )


def test_the_lock_is_reentrant():
    """Public methods call other public methods, so a plain Lock would deadlock.

    Pinned because swapping `RLock` for `Lock` looks like a harmless tightening
    and would hang the server rather than fail a test elsewhere.
    """
    lock = AgenticSearchStore(":memory:")._lock
    assert lock.acquire(timeout=1)
    try:
        assert lock.acquire(timeout=1), "store lock is not reentrant"
        lock.release()
    finally:
        lock.release()
