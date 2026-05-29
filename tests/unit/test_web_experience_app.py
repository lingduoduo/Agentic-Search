from __future__ import annotations

from fastapi.testclient import TestClient

from src.context.models import AnswerGenerationResult
from src.context.models import ChatMessage
from src.context.models import ContextDocument
from src.context.models import PromptBundle
from src.context.models import SearchContextBundle
from src.db import AgenticSearchStore
from src.hooks import HookConfig
from src.hooks import HookPoint
from src.hooks import HookRegistry
from src.servers.web.app import SearchExperienceSettings
from src.servers.web.app import create_web_app


def _answer_result(question: str) -> AnswerGenerationResult:
    context = SearchContextBundle(
        query=question,
        documents=[
            ContextDocument(
                id="D1",
                title="Deployment Guide",
                content="Use the retrieval server before starting the web app.",
                url="https://example.test/deploy",
                score=0.91,
                metadata={"source_type": "docs"},
            )
        ],
    )
    return AnswerGenerationResult(
        answer="[D1] Start retrieval first, then open the web app.",
        citations=["D1"],
        context=context,
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_web_app_serves_browser_experience(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Agentic Search" in response.text
    assert "/api/agent" in client.get("/assets/app.js").text
    assert "text/css" in client.get("/assets/app.css").headers["content-type"]


def test_agent_endpoint_runs_pipeline_and_persists_chat(monkeypatch, tmp_path):
    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        assert filters is None
        assert question == "How do I deploy?"
        assert chat_history == []
        assert search_url == "http://search.test/retrieve"
        assert top_k == 3
        return _answer_result(question)

    monkeypatch.setattr(
        "src.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    app = create_web_app(store=store)
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={
            "query": "How do I deploy?",
            "search_url": "http://search.test/retrieve",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "[D1] Start retrieval first, then open the web app."
    assert data["documents"][0]["title"] == "Deployment Guide"
    assert [message["role"] for message in data["messages"]] == ["user", "assistant"]

    session = client.get(f"/api/sessions/{data['session_id']}").json()
    assert [message["content"] for message in session["messages"]] == [
        "How do I deploy?",
        "[D1] Start retrieval first, then open the web app.",
    ]
    store.close()


def test_agent_endpoint_reuses_existing_session_history(monkeypatch, tmp_path):
    observed_history: list[ChatMessage] = []

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        del filters
        observed_history.extend(chat_history or [])
        return _answer_result(question)

    monkeypatch.setattr(
        "src.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Deployment"}).json()
    client.post(
        "/api/agent",
        json={"query": "First question", "session_id": session["id"]},
    )

    response = client.post(
        "/api/agent",
        json={"query": "Follow up", "session_id": session["id"]},
    )

    assert response.status_code == 200
    assert [message.role for message in observed_history] == ["user", "assistant"]


def test_agent_endpoint_runs_query_processing_hook(monkeypatch, tmp_path):
    class FakeHookResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"query": "rewritten deploy question", "metadata": {"source": "hook"}}'

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
    ) -> AnswerGenerationResult:
        del llm, chat_history, search_url, top_k, filters
        assert question == "rewritten deploy question"
        return _answer_result(question)

    monkeypatch.setattr(
        "src.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    monkeypatch.setattr(
        "src.hooks.executor.urllib.request.urlopen",
        lambda request, timeout: FakeHookResponse(),
    )
    registry = HookRegistry(
        [
            HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
            )
        ]
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        hook_registry=registry,
    )
    client = TestClient(app)

    response = client.post("/api/agent", json={"query": "How do I deploy?"})

    assert response.status_code == 200
    session = client.get(f"/api/sessions/{response.json()['session_id']}").json()
    assert session["messages"][0]["content"] == "rewritten deploy question"
    assert registry.execution_log[-1].is_success is True
