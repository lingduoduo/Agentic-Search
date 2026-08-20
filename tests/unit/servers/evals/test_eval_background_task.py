"""The queued eval must survive long enough to actually run.

`eval_run_ack` dispatches the search with `asyncio.ensure_future` and kept no
reference to the resulting task. The event loop holds only a *weak* reference,
so a pending background eval could be garbage-collected mid-flight — and since
the route's whole observable output is the log line it writes on completion,
losing the task loses the feature silently.

The GC race is not deterministically reproducible, which is exactly what makes
it dangerous. These tests pin the mechanism that prevents it instead: the task
is strongly referenced while it runs, released once it finishes, and the work
reaches completion.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.internal.configs import AppSettings
from src.internal.servers.evals import api as evals_api


def ack_endpoint():
    """Pull `eval_run_ack` back out of the router closure."""
    router = evals_api.create_evals_router(
        AppSettings(), require_admin=lambda: SimpleNamespace(id="admin")
    )
    return next(
        route.endpoint for route in router.routes if route.name == "eval_run_ack"
    )


def a_query():
    return evals_api.EvalConfigurationOptions(query="what is FAISS?")


async def drain_background_tasks() -> None:
    """Await whatever is pending, without looping on the set itself.

    Looping until the set empties would hang forever if the done-callback that
    empties it ever regressed, and a hanging suite is worse than a red one.
    """
    pending = tuple(evals_api._background_tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)  # let the done-callbacks run


@pytest.fixture(autouse=True)
def isolate_background_tasks():
    """The task set is module state; do not let one test observe another's."""
    evals_api._background_tasks.clear()
    yield
    evals_api._background_tasks.clear()


@pytest.mark.asyncio
async def test_the_queued_eval_actually_runs() -> None:
    searched: list[str] = []

    async def record_search(query: str, **_kwargs):
        searched.append(query)
        return Mock(results=[])

    with patch.object(evals_api, "run_expanded_search", record_search):
        response = await ack_endpoint()(a_query(), _=None)
        await drain_background_tasks()

    assert response.success is True
    assert searched == ["what is FAISS?"]


@pytest.mark.asyncio
async def test_the_task_is_strongly_referenced_while_it_is_still_running() -> None:
    release = asyncio.Event()

    async def blocked_search(_query: str, **_kwargs):
        await release.wait()
        return Mock(results=[])

    with patch.object(evals_api, "run_expanded_search", blocked_search):
        await ack_endpoint()(a_query(), _=None)
        await asyncio.sleep(0)

        # The loop only weak-references tasks; something else has to hold this
        # one or it can be collected before `release` is ever set.
        assert evals_api._background_tasks, "pending eval task is unreferenced"

        release.set()
        await drain_background_tasks()


@pytest.mark.asyncio
async def test_a_finished_task_is_released_so_the_set_cannot_grow_forever() -> None:
    async def instant_search(_query: str, **_kwargs):
        return Mock(results=[])

    with patch.object(evals_api, "run_expanded_search", instant_search):
        for _ in range(3):
            await ack_endpoint()(a_query(), _=None)
        await drain_background_tasks()

    assert evals_api._background_tasks == set()


@pytest.mark.asyncio
async def test_a_failing_eval_is_logged_and_does_not_escape() -> None:
    async def failing_search(_query: str, **_kwargs):
        raise RuntimeError("retrieval server down")

    with (
        patch.object(evals_api, "run_expanded_search", failing_search),
        patch.object(evals_api.logger, "error") as logged,
    ):
        await ack_endpoint()(a_query(), _=None)
        await drain_background_tasks()

    assert logged.called
    assert evals_api._background_tasks == set()
