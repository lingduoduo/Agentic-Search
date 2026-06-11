"""LangGraph-backed BaseAgent: streaming via AgentQueueManager."""

from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Iterator, Optional
from uuid import UUID

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
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


class LangGraphAgentState(TypedDict, total=False):
    """Minimal LangGraph state dict threaded through the compiled graph."""

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
    """Raised when the agent graph was not built or is unavailable."""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent(BaseModel):
    """Abstract LangGraph agent with streaming output via :class:`AgentQueueManager`.

    Subclasses must implement :meth:`_build_agent`, which returns a
    ``CompiledStateGraph``.  Graph nodes publish :class:`AgentThought` events to
    ``self.queue_manager`` during execution; :meth:`stream` starts the graph in a
    background thread and yields those events; :meth:`invoke` collects them into
    an :class:`AgentResult`.

    Example::

        class MyAgent(BaseAgent):
            def _build_agent(self) -> CompiledStateGraph:
                def call_model(state):
                    tid = state["task_id"]
                    self.queue_manager.publish(
                        tid,
                        AgentThought(id=uuid.uuid4(), task_id=tid,
                                     event=QueueEvent.AGENT_MESSAGE, answer="hi"),
                    )
                    self.queue_manager.publish(
                        tid,
                        AgentThought(id=uuid.uuid4(), task_id=tid,
                                     event=QueueEvent.AGENT_END),
                    )
                    return state

                graph = StateGraph(LangGraphAgentState)
                graph.add_node("model", call_model)
                graph.set_entry_point("model")
                graph.add_edge("model", END)
                return graph.compile()

        agent = MyAgent(llm=llm, agent_config=config, redis_client=redis)
        result = agent.invoke({"messages": [HumanMessage(content="hello")]})
    """

    llm: BaseLanguageModel
    agent_config: AgentConfig
    redis_client: TenantRedisClient

    _agent: CompiledStateGraph | None = PrivateAttr(default=None)
    _queue_manager: AgentQueueManager | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._agent = self._build_agent()
        self._queue_manager = AgentQueueManager(
            user_id=self.agent_config.user_id,
            invoke_from=self.agent_config.invoke_from,
            redis_client=self.redis_client,
        )

    @abstractmethod
    def _build_agent(self) -> CompiledStateGraph:
        """Construct and return the compiled LangGraph state graph."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: LangGraphAgentState,
        config: Optional[RunnableConfig] = None,
    ) -> AgentResult:
        """Run the agent to completion and return aggregated results."""
        query = ""
        messages = input.get("messages", [])
        if messages:
            first = messages[0]
            query = first.content if hasattr(first, "content") else str(first)

        result = AgentResult(query=query)
        accumulated: dict[str, AgentThought] = {}

        for thought in self.stream(input, config):
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

    def stream(
        self,
        input: LangGraphAgentState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AgentThought]:
        """Run the graph in a background thread and yield streaming events."""
        if self._agent is None:
            raise AgentBuildError("Agent graph is not built. Check _build_agent().")
        if self._queue_manager is None:
            raise AgentBuildError("Queue manager not initialised.")

        if "task_id" not in input:
            input["task_id"] = uuid.uuid4()
        input.setdefault("history", [])
        input.setdefault("iteration_count", 0)

        thread = Thread(target=self._agent.invoke, args=(input,), daemon=True)
        thread.start()

        yield from self._queue_manager.listen(input["task_id"])

    @property
    def queue_manager(self) -> AgentQueueManager:
        if self._queue_manager is None:
            raise AgentBuildError("Queue manager not initialised.")
        return self._queue_manager
