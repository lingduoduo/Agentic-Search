"""Per-task agent queue manager with Redis-backed stop/ownership control."""

from __future__ import annotations

import queue
import time
import uuid
from collections.abc import Generator
from enum import Enum
from queue import Queue
from typing import Any
from uuid import UUID

from src.internal.configs.constants import InvokeFrom
from src.internal.servers.redis.tenant_redis_client import TenantRedisClient

_LISTEN_TIMEOUT_SECONDS = 600
_PING_INTERVAL_SECONDS = 10
_TASK_BELONG_TTL = 1800
_TASK_STOPPED_TTL = 600

# Callers whose tasks are keyed under "account" rather than "end-user"
_ACCOUNT_INVOKE_ORIGINS = {
    InvokeFrom.WEB_APP,
    InvokeFrom.DEBUGGER,
    InvokeFrom.ASSISTANT_AGENT,
}


class QueueEvent(str, Enum):
    PING = "ping"
    STOP = "stop"
    ERROR = "error"
    TIMEOUT = "timeout"
    AGENT_END = "agent_end"
    AGENT_MESSAGE = "agent_message"
    AGENT_THOUGHT = "agent_thought"


class AgentThought:
    """Event placed on the task queue."""

    __slots__ = (
        "id",
        "task_id",
        "event",
        "observation",
        "thought",
        "answer",
        "message",
        "latency",
    )

    def __init__(
        self,
        *,
        id: UUID,
        task_id: UUID,
        event: QueueEvent,
        observation: str | None = None,
        thought: str = "",
        answer: str = "",
        message: list | None = None,
        latency: float = 0.0,
    ) -> None:
        self.id = id
        self.task_id = task_id
        self.event = event
        self.observation = observation
        self.thought = thought
        self.answer = answer
        self.message = message if message is not None else []
        self.latency = latency

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "event": self.event.value,
            "observation": self.observation,
            "thought": self.thought,
            "answer": self.answer,
            "latency": self.latency,
        }


class AgentQueueManager:
    """Per-user, per-task in-memory queue with Redis-backed stop and ownership control.

    Usage::

        redis = get_shared_redis_client()
        manager = AgentQueueManager(user_id=uid, invoke_from=InvokeFrom.WEB_APP, redis_client=redis)

        # Producer (agent loop):
        manager.publish(task_id, AgentThought(id=..., task_id=task_id, event=QueueEvent.AGENT_MESSAGE))

        # Consumer (SSE endpoint):
        for thought in manager.listen(task_id):
            yield thought.to_dict()

        # External stop (e.g. DELETE /tasks/<task_id>):
        AgentQueueManager.set_stop_flag(task_id, invoke_from, user_id, redis_client=redis)
    """

    def __init__(
        self,
        *,
        user_id: UUID,
        invoke_from: InvokeFrom,
        redis_client: TenantRedisClient,
    ) -> None:
        self.user_id = user_id
        self.invoke_from = invoke_from
        self._redis = redis_client
        self._queues: dict[str, Queue[AgentThought | None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, task_id: UUID) -> Generator[AgentThought, None, None]:
        """Block-read the task queue and yield events until done or timed out."""
        start_time = time.time()
        last_ping_bucket = 0

        while True:
            try:
                item = self._get_queue(task_id).get(timeout=1)
                if item is None:
                    break
                yield item
            except queue.Empty:
                pass

            elapsed = time.time() - start_time
            ping_bucket = int(elapsed // _PING_INTERVAL_SECONDS)

            if ping_bucket > last_ping_bucket:
                self.publish(task_id, self._event(task_id, QueueEvent.PING))
                last_ping_bucket = ping_bucket

            if elapsed >= _LISTEN_TIMEOUT_SECONDS:
                # Yield directly: publish() enqueues onto this same queue, which
                # we never re-read because we break immediately after.
                yield self._event(task_id, QueueEvent.TIMEOUT)
                break

            if self._is_stopped(task_id):
                yield self._event(task_id, QueueEvent.STOP)
                break

    def stop_listen(self, task_id: UUID) -> None:
        """Unblock the listener by placing a sentinel None on the queue."""
        self._get_queue(task_id).put(None)

    def publish(self, task_id: UUID, thought: AgentThought) -> None:
        """Put an event on the queue; automatically closes it on terminal events."""
        self._get_queue(task_id).put(thought)
        if thought.event in {
            QueueEvent.STOP,
            QueueEvent.ERROR,
            QueueEvent.TIMEOUT,
            QueueEvent.AGENT_END,
        }:
            self.stop_listen(task_id)

    def publish_error(self, task_id: UUID, error: Exception | str) -> None:
        """Convenience wrapper that publishes a QueueEvent.ERROR thought."""
        self.publish(
            task_id,
            AgentThought(
                id=uuid.uuid4(),
                task_id=task_id,
                event=QueueEvent.ERROR,
                observation=str(error),
            ),
        )

    # ------------------------------------------------------------------
    # Stop-flag control (classmethod so callers don't need an instance)
    # ------------------------------------------------------------------

    @classmethod
    def set_stop_flag(
        cls,
        task_id: UUID,
        invoke_from: InvokeFrom,
        user_id: UUID,
        *,
        redis_client: TenantRedisClient,
    ) -> None:
        """Write the stop flag for a task if the caller owns it."""
        belong_key = cls._belong_cache_key(task_id)
        result = redis_client.get(belong_key)
        if result is None:
            return

        user_prefix = cls._user_prefix(invoke_from)
        if result.decode() != f"{user_prefix}-{user_id}":
            return

        redis_client.setex(cls._stopped_cache_key(task_id), _TASK_STOPPED_TTL, 1)

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------

    @classmethod
    def _belong_cache_key(cls, task_id: UUID) -> str:
        return f"task_belong:{task_id}"

    @classmethod
    def _stopped_cache_key(cls, task_id: UUID) -> str:
        return f"task_stopped:{task_id}"

    @staticmethod
    def _user_prefix(invoke_from: InvokeFrom) -> str:
        return "account" if invoke_from in _ACCOUNT_INVOKE_ORIGINS else "end-user"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_queue(self, task_id: UUID) -> Queue[AgentThought | None]:
        key = str(task_id)
        q = self._queues.get(key)
        if q is None:
            user_prefix = self._user_prefix(self.invoke_from)
            self._redis.setex(
                self._belong_cache_key(task_id),
                _TASK_BELONG_TTL,
                f"{user_prefix}-{self.user_id}",
            )
            q = Queue()
            self._queues[key] = q
        return q

    def _is_stopped(self, task_id: UUID) -> bool:
        return self._redis.get(self._stopped_cache_key(task_id)) is not None

    @staticmethod
    def _event(task_id: UUID, event: QueueEvent) -> AgentThought:
        return AgentThought(id=uuid.uuid4(), task_id=task_id, event=event)
