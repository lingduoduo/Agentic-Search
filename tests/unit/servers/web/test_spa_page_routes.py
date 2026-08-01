"""Page paths must serve the SPA shell, and only the page paths.

The frontend routes /assist, /search, /chat and /tools client-side. A refresh
or a pasted link hits the backend directly, so each path has to return the app
shell rather than a 404 — without shadowing the API routes that share a prefix.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.servers.web.app import create_web_app

PAGE_PATHS = ["/assist", "/search", "/chat", "/tools"]


@pytest.fixture
def client() -> TestClient:
    # No `with`: the lifespan would load SEARCH_AGENT_MODEL, which this does not need.
    return TestClient(create_web_app(store=AgenticSearchStore(":memory:")))


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_page_path_serves_the_app_shell(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.text == client.get("/").text


def test_api_paths_under_a_page_prefix_still_reach_the_api(client: TestClient) -> None:
    """The page route must not shadow /chat/* — it is a sibling, not a parent."""
    response = client.post("/chat/create-chat-session", json={})

    assert response.status_code == 200
    assert "chat_session_id" in response.json()


def test_an_unknown_path_is_still_a_404(client: TestClient) -> None:
    """Explicit page routes, not a catch-all: a typo must not return HTML."""
    response = client.get("/definitely-not-a-page")

    assert response.status_code == 404
