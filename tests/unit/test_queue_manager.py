"""Unit tests for AgentQueueManager."""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock


from src.internal.chat import queue_manager as qm
from src.internal.chat.queue_manager import AgentQueueManager, AgentThought, QueueEvent
from src.internal.configs.constants import InvokeFrom


def _redis() -> MagicMock:
    """Minimal Redis mock: get() returns None by default (task not stopped)."""
    r = MagicMock()
    r.get.return_value = None
    return r


def _manager(invoke_from: InvokeFrom = InvokeFrom.WEB_APP) -> AgentQueueManager:
    return AgentQueueManager(
        user_id=uuid.uuid4(),
        invoke_from=invoke_from,
        redis_client=_redis(),
    )


# ---------------------------------------------------------------------------
# Basic publish / listen
# ---------------------------------------------------------------------------


def test_listen_yields_published_thoughts():
    mgr = _manager()
    task_id = uuid.uuid4()

    def _produce():
        mgr.publish(
            task_id,
            AgentThought(
                id=uuid.uuid4(),
                task_id=task_id,
                event=QueueEvent.AGENT_MESSAGE,
                observation="hello",
            ),
        )
        mgr.publish(
            task_id,
            AgentThought(id=uuid.uuid4(), task_id=task_id, event=QueueEvent.AGENT_END),
        )

    t = threading.Thread(target=_produce)
    t.start()

    events = [item.event for item in mgr.listen(task_id)]
    t.join()

    assert QueueEvent.AGENT_MESSAGE in events
    assert QueueEvent.AGENT_END in events


def test_terminal_event_closes_queue():
    """Publishing a terminal event should put a sentinel so listen() exits."""
    mgr = _manager()
    task_id = uuid.uuid4()

    mgr.publish(
        task_id,
        AgentThought(id=uuid.uuid4(), task_id=task_id, event=QueueEvent.AGENT_END),
    )

    events = list(mgr.listen(task_id))
    assert events[-1].event == QueueEvent.AGENT_END


def test_publish_error_emits_error_event():
    mgr = _manager()
    task_id = uuid.uuid4()

    mgr.publish_error(task_id, "something broke")

    events = list(mgr.listen(task_id))
    assert events[0].event == QueueEvent.ERROR
    assert events[0].observation == "something broke"


# ---------------------------------------------------------------------------
# Stop flag
# ---------------------------------------------------------------------------


def test_set_stop_flag_writes_redis_key():
    redis = _redis()
    user_id = uuid.uuid4()
    task_id = uuid.uuid4()
    invoke_from = InvokeFrom.WEB_APP

    # Simulate ownership record already set
    redis.get.return_value = f"account-{user_id}".encode()

    AgentQueueManager.set_stop_flag(task_id, invoke_from, user_id, redis_client=redis)

    redis.setex.assert_called_once()
    key_arg = redis.setex.call_args[0][0]
    assert "task_stopped" in key_arg


def test_set_stop_flag_ignores_unknown_task():
    redis = _redis()
    redis.get.return_value = None  # task never started

    AgentQueueManager.set_stop_flag(
        uuid.uuid4(), InvokeFrom.WEB_APP, uuid.uuid4(), redis_client=redis
    )

    redis.setex.assert_not_called()


def test_set_stop_flag_ignores_wrong_owner():
    redis = _redis()
    task_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    redis.get.return_value = f"account-{owner_id}".encode()

    AgentQueueManager.set_stop_flag(
        task_id, InvokeFrom.WEB_APP, other_id, redis_client=redis
    )

    redis.setex.assert_not_called()


# ---------------------------------------------------------------------------
# Self-detected stop / timeout are delivered to the consumer
# ---------------------------------------------------------------------------


def test_listen_yields_stop_event_when_stopped(monkeypatch):
    """A stop detected inside listen() must be yielded, not just enqueued."""
    mgr = _manager()
    task_id = uuid.uuid4()
    monkeypatch.setattr(mgr, "_is_stopped", lambda _tid: True)

    events = list(mgr.listen(task_id))

    assert events[-1].event == QueueEvent.STOP


def test_listen_yields_timeout_event_when_timed_out(monkeypatch):
    """A timeout detected inside listen() must be yielded, not just enqueued."""
    mgr = _manager()
    task_id = uuid.uuid4()
    monkeypatch.setattr(qm, "_LISTEN_TIMEOUT_SECONDS", 0)

    events = list(mgr.listen(task_id))

    assert events[-1].event == QueueEvent.TIMEOUT


# ---------------------------------------------------------------------------
# User prefix / ownership key
# ---------------------------------------------------------------------------


def test_account_invoke_origins_use_account_prefix():
    for invoke_from in (
        InvokeFrom.WEB_APP,
        InvokeFrom.DEBUGGER,
        InvokeFrom.ASSISTANT_AGENT,
    ):
        assert AgentQueueManager._user_prefix(invoke_from) == "account"


def test_service_api_uses_end_user_prefix():
    assert AgentQueueManager._user_prefix(InvokeFrom.SERVICE_API) == "end-user"


# ---------------------------------------------------------------------------
# AgentThought serialisation
# ---------------------------------------------------------------------------


def test_agent_thought_to_dict():
    tid = uuid.uuid4()
    thought = AgentThought(
        id=uuid.uuid4(), task_id=tid, event=QueueEvent.PING, observation=None
    )
    d = thought.to_dict()
    assert d["event"] == "ping"
    assert d["task_id"] == str(tid)
    assert d["observation"] is None
