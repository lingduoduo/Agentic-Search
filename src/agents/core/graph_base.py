"""Pydantic-based BaseAgent with streaming via AgentQueueManager.

Replaces the previous LangGraph-backed implementation with a simpler design:
subclasses implement ``run(state)`` directly instead of building a compiled
state graph.  Threading and event delivery are handled by this base class.

    class MyAgent(BaseAgent):
        def run(self, state: AgentState) -> None:
            tid = state["task_id"]
            self.queue_manager.publish(
                tid,
                AgentThought(id=uuid4(), task_id=tid,
                             event=QueueEvent.AGENT_MESSAGE, answer="hi"),
            )
            self.queue_manager.publish(
                tid,
                AgentThought(id=uuid4(), task_id=tid, event=QueueEvent.AGENT_END),
            )

    agent = MyAgent(llm=my_llm, agent_config=config, redis_client=redis)
    result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Iterator
from uuid import UUID

from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

from src.internal.chat.queue_manager import AgentQueueManager, AgentThought, QueueEvent
from src.internal.configs.constants import InvokeFrom
from src.internal.servers.redis.tenant_redis_client import TenantRedisClient


# ---------------------------------------------------------------------------
# Configuration and state types
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Runtime configuration passed to every BaseAgent."""

    user_id: UUID
    invoke_from: InvokeFrom = InvokeFrom.WEB_APP

    model_config = {"arbitrary_types_allowed": True}


class AgentState(TypedDict, total=False):
    """Mutable state dict threaded through one agent invocation."""

    task_id: UUID
    messages: list[Any]
    history: list[Any]
    iteration_count: int


@dataclass
class AgentResult:
    """Aggregated output from a blocking :meth:`BaseAgent.invoke` call."""

    query: str
    answer: str = ""
    status: QueueEvent | None = None
    error: str = ""
    agent_thoughts: list[AgentThought] = field(default_factory=list)
    message: list[Any] = field(default_factory=list)
    latency: float = 0.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentBuildError(RuntimeError):
    """Raised when the agent is not ready to run."""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent(BaseModel):
    """Abstract Pydantic agent with streaming output via :class:`AgentQueueManager`.

    Subclasses implement :meth:`run`, which executes the agent logic and
    publishes :class:`AgentThought` events to ``self.queue_manager``.
    :meth:`stream` starts :meth:`run` in a background thread and yields
    those events; :meth:`invoke` collects them into an :class:`AgentResult`.

    ``llm`` accepts any language model object — it is typed ``Any`` to avoid
    coupling to a specific LLM SDK.
    """

    llm: Any
    agent_config: AgentConfig
    redis_client: TenantRedisClient

    _queue_manager: AgentQueueManager | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._queue_manager = AgentQueueManager(
            user_id=self.agent_config.user_id,
            invoke_from=self.agent_config.invoke_from,
            redis_client=self.redis_client,
        )

    @abstractmethod
    def run(self, state: AgentState) -> None:
        """Execute agent logic and publish events to ``self.queue_manager``.

        Must publish at least one terminal event
        (:attr:`QueueEvent.AGENT_END`, :attr:`QueueEvent.ERROR`, or
        :attr:`QueueEvent.STOP`) so :meth:`stream` can exit cleanly.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, input: AgentState) -> AgentResult:
        """Run the agent to completion and return aggregated results."""
        query = ""
        messages = input.get("messages", [])
        if messages:
            first = messages[0]
            query = (
                first.get("content", "")
                if isinstance(first, dict)
                else getattr(first, "content", str(first))
            )

        result = AgentResult(query=query)
        accumulated: dict[str, AgentThought] = {}

        for thought in self.stream(input):
            eid = str(thought.id)

            if thought.event == QueueEvent.PING:
                continue

            if thought.event == QueueEvent.AGENT_MESSAGE:
                if eid not in accumulated:
                    accumulated[eid] = thought
                else:
                    existing = accumulated[eid]
                    existing.thought += thought.thought
                    existing.answer += thought.answer
                    existing.latency = thought.latency
                result.answer += thought.answer
            else:
                accumulated[eid] = thought
                if thought.event in {
                    QueueEvent.STOP,
                    QueueEvent.TIMEOUT,
                    QueueEvent.ERROR,
                }:
                    result.status = thought.event
                    result.error = (
                        thought.observation or ""
                        if thought.event == QueueEvent.ERROR
                        else ""
                    )

        result.agent_thoughts = list(accumulated.values())
        result.message = next(
            (
                t.message
                for t in accumulated.values()
                if t.event == QueueEvent.AGENT_MESSAGE
            ),
            [],
        )
        result.latency = sum(t.latency for t in accumulated.values())
        return result

    def stream(self, input: AgentState) -> Iterator[AgentThought]:
        """Run :meth:`run` in a background thread and yield streaming events."""
        if self._queue_manager is None:
            raise AgentBuildError("Queue manager not initialised.")

        if "task_id" not in input:
            input["task_id"] = uuid.uuid4()
        input.setdefault("history", [])
        input.setdefault("iteration_count", 0)

        thread = Thread(target=self.run, args=(input,), daemon=True)
        thread.start()

        yield from self._queue_manager.listen(input["task_id"])

    @property
    def queue_manager(self) -> AgentQueueManager:
        if self._queue_manager is None:
            raise AgentBuildError("Queue manager not initialised.")
        return self._queue_manager
