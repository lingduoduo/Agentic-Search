"""Unit tests for BaseAgent (LangGraph-backed streaming agent)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from src.agents.graph_base import AgentBuildError, AgentConfig, AgentResult, BaseAgent
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


def _llm_mock() -> MagicMock:
    return MagicMock()


def _make_agent(publish_fn) -> BaseAgent:
    """Build a concrete BaseAgent whose graph calls publish_fn(state, qm).

    Uses model_construct() to skip Pydantic validation so MagicMock objects
    can stand in for llm and redis_client in unit tests.
    """

    class _Agent(BaseAgent):
        def _build_agent(self):
            mock_graph = MagicMock()

            def _invoke(state):
                publish_fn(state, self.queue_manager)

            mock_graph.invoke.side_effect = _invoke
            return mock_graph

    cfg = _config()
    redis = _redis_mock()
    agent = _Agent.model_construct(
        llm=_llm_mock(), agent_config=cfg, redis_client=redis
    )
    # model_construct skips model_post_init; wire private attrs manually
    agent._agent = agent._build_agent()
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
    def publish(state, qm):
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

    agent = _make_agent(publish)
    events = [t.event for t in agent.stream({"messages": []})]
    assert QueueEvent.AGENT_MESSAGE in events
    assert QueueEvent.AGENT_END in events


def test_stream_injects_task_id_if_missing():
    seen_task_ids: list = []

    def publish(state, qm):
        seen_task_ids.append(state["task_id"])
        tid = state["task_id"]
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(publish)
    list(agent.stream({"messages": []}))
    assert len(seen_task_ids) == 1
    assert seen_task_ids[0] is not None


def test_stream_uses_provided_task_id():
    task_id = uuid.uuid4()
    seen: list = []

    def publish(state, qm):
        seen.append(state["task_id"])
        qm.publish(
            task_id,
            AgentThought(id=uuid.uuid4(), task_id=task_id, event=QueueEvent.AGENT_END),
        )

    agent = _make_agent(publish)
    list(agent.stream({"task_id": task_id, "messages": []}))
    assert seen[0] == task_id


# ---------------------------------------------------------------------------
# invoke() tests
# ---------------------------------------------------------------------------


def test_invoke_accumulates_answer():
    def publish(state, qm):
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

    agent = _make_agent(publish)

    class _Msg:
        content = "what?"

    result = agent.invoke({"messages": [_Msg()]})
    assert isinstance(result, AgentResult)
    assert result.query == "what?"
    assert result.answer == "foobar"


def test_invoke_skips_ping_events():
    def publish(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.PING)
        )
        qm.publish(
            tid, AgentThought(id=uuid.uuid4(), task_id=tid, event=QueueEvent.AGENT_END)
        )

    agent = _make_agent(publish)
    result = agent.invoke({"messages": []})
    assert all(t.event != QueueEvent.PING for t in result.agent_thoughts)


def test_invoke_sets_status_on_error():
    def publish(state, qm):
        tid = state["task_id"]
        qm.publish(
            tid,
            AgentThought(
                id=uuid.uuid4(), task_id=tid, event=QueueEvent.ERROR, observation="boom"
            ),
        )

    agent = _make_agent(publish)
    result = agent.invoke({"messages": []})
    assert result.status == QueueEvent.ERROR
    assert result.error == "boom"


def test_invoke_sums_latency():
    def publish(state, qm):
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

    agent = _make_agent(publish)
    result = agent.invoke({"messages": []})
    assert abs(result.latency - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_stream_raises_if_agent_not_built():
    class _BrokenAgent(BaseAgent):
        def _build_agent(self):
            return None  # simulate failure

    cfg = _config()
    redis = _redis_mock()
    agent = _BrokenAgent.model_construct(
        llm=_llm_mock(), agent_config=cfg, redis_client=redis
    )
    agent._agent = None
    agent._queue_manager = AgentQueueManager(
        user_id=cfg.user_id, invoke_from=cfg.invoke_from, redis_client=redis
    )
    with pytest.raises(AgentBuildError):
        list(agent.stream({"messages": []}))
