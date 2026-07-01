"""Unit tests for the Pydantic-based BaseAgent."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from src.agents.core.graph_base import (
    AgentBuildError,
    AgentConfig,
    AgentResult,
    BaseAgent,
    AgentState,
)
from src.internal.chat.queue_manager import AgentQueueManager, AgentThought, QueueEvent
from src.internal.configs.constants import InvokeFrom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redis_mock() -> MagicMock:
    r = MagicMock()
    r.get.return_value = None
    return r


def _config() -> AgentConfig:
    return AgentConfig(user_id=uuid.uuid4(), invoke_from=InvokeFrom.WEB_APP)


def _make_agent(run_fn) -> BaseAgent:
    """Build a concrete BaseAgent whose run() calls run_fn(state, qm)."""

    class _Agent(BaseAgent):
        def run(self, state: AgentState) -> None:
            run_fn(state, self.queue_manager)

    cfg = _config()
    redis = _redis_mock()
    agent = _Agent.model_construct(
        llm=MagicMock(), agent_config=cfg, redis_client=redis
    )
    agent._queue_manager = AgentQueueManager(
        user_id=cfg.user_id,
        invoke_from=cfg.invoke_from,
        redis_client=redis,
    )
    return agent


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


def test_stream_yields_events():
    def run(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(),
                task_id=tid,
                event=QueueEvent.AGENT_MESSAGE,
                answer="hello",
            ),
        )
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(run)
    events = [t.event for t in agent.stream({"messages": []})]
    assert QueueEvent.AGENT_MESSAGE in events
    assert QueueEvent.AGENT_END in events


def test_stream_injects_task_id_if_missing():
    seen: list = []

    def run(state, qm):
        seen.append(state["task_id"])
        tid = state["task_id"]
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(run)
    list(agent.stream({"messages": []}))
    assert len(seen) == 1
    assert seen[0] is not None


def test_stream_uses_provided_task_id():
    task_id = uuid.uuid4()
    seen: list = []

    def run(state, qm):
        seen.append(state["task_id"])
        qm.publish(
            task_id,
            AgentThought(id=uuid.uuid4(), task_id=task_id, event=QueueEvent.AGENT_END),
        )

    agent = _make_agent(run)
    list(agent.stream({"task_id": task_id, "messages": []}))
    assert seen[0] == task_id


# ---------------------------------------------------------------------------
# invoke() tests
# ---------------------------------------------------------------------------


def test_invoke_accumulates_answer():
    def run(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(),
                task_id=tid,
                event=QueueEvent.AGENT_MESSAGE,
                answer="foo",
            ),
        )
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(),
                task_id=tid,
                event=QueueEvent.AGENT_MESSAGE,
                answer="bar",
            ),
        )
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(run)
    result = agent.invoke({"messages": [{"role": "user", "content": "what?"}]})
    assert isinstance(result, AgentResult)
    assert result.query == "what?"
    assert result.answer == "foobar"


def test_invoke_skips_ping_events():
    def run(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.PING)
        )
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(run)
    result = agent.invoke({"messages": []})
    assert all(t.event != QueueEvent.PING for t in result.agent_thoughts)


def test_invoke_sets_status_on_error():
    def run(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(), task_id=tid, event=QueueEvent.ERROR, observation="boom"
            ),
        )

    agent = _make_agent(run)
    result = agent.invoke({"messages": []})
    assert result.status == QueueEvent.ERROR
    assert result.error == "boom"


def test_invoke_sums_latency():
    def run(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(),
                task_id=tid,
                event=QueueEvent.AGENT_MESSAGE,
                latency=0.3,
            ),
        )
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END, latency=0.1
            ),
        )

    agent = _make_agent(run)
    result = agent.invoke({"messages": []})
    assert abs(result.latency - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_stream_raises_if_queue_manager_not_initialised():
    class _BrokenAgent(BaseAgent):
        def run(self, state: AgentState) -> None:
            pass

    cfg = _config()
    agent = _BrokenAgent.model_construct(
        llm=MagicMock(), agent_config=cfg, redis_client=MagicMock()
    )
    agent._queue_manager = None
    with pytest.raises(AgentBuildError):
        list(agent.stream({"messages": []}))
