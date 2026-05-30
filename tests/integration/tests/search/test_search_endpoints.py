# ruff: noqa: F821
"""Integration tests for search and chat endpoints.

These tests exercise the endpoints that the local SQLite+FAISS backend
actually implements, rather than the full Danswer infrastructure.

Covered:
- POST /search/search-flow-classification
- POST /search/send-search-message  (non-streaming)
- GET  /search/search-history
- GET  /me  (auth smoke-test)
"""

import requests

from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.test_models import DATestUser


def test_me_returns_current_user(admin_user: DATestUser) -> None:
    """Authenticated GET /me returns the logged-in user's id and role."""
    resp = requests.get(
        f"{API_SERVER_URL}/me",
        headers=admin_user.headers,
        cookies=admin_user.cookies,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["id"] == admin_user.id
    assert data["email"] == admin_user.email
    assert data["role"] == "admin"
    assert data["is_active"] is True


def test_me_unauthenticated_returns_401() -> None:
    """GET /me without credentials returns 401."""
    resp = requests.get(f"{API_SERVER_URL}/me")
    assert resp.status_code == 401


def test_search_flow_classification_short_query(admin_user: DATestUser) -> None:
    """Short queries are classified as search flow."""
    resp = requests.post(
        f"{API_SERVER_URL}/search/search-flow-classification",
        json={"user_query": "what is RAG?"},
        headers=admin_user.headers,
        cookies=admin_user.cookies,
    )
    assert resp.status_code == 200, resp.text
    assert "is_search_flow" in resp.json()


def test_search_flow_classification_long_query(admin_user: DATestUser) -> None:
    """Queries longer than 200 chars are always classified as chat (not search)."""
    long_query = "explain " + "in great detail " * 20
    resp = requests.post(
        f"{API_SERVER_URL}/search/search-flow-classification",
        json={"user_query": long_query},
        headers=admin_user.headers,
        cookies=admin_user.cookies,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["is_search_flow"] is False


def test_send_search_message_returns_response(admin_user: DATestUser) -> None:
    """POST /search/send-search-message returns a structured response (stream=False)."""
    resp = requests.post(
        f"{API_SERVER_URL}/search/send-search-message",
        json={
            "search_query": "test query",
            "num_hits": 3,
            "stream": False,
            "run_query_expansion": False,
        },
        headers=admin_user.headers,
        cookies=admin_user.cookies,
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "search_docs" in data
    assert "all_executed_queries" in data
    assert isinstance(data["search_docs"], list)


def test_search_history_returns_list(admin_user: DATestUser) -> None:
    """GET /search/search-history returns a list (may be empty on a fresh server)."""
    resp = requests.get(
        f"{API_SERVER_URL}/search/search-history",
        headers=admin_user.headers,
        cookies=admin_user.cookies,
    )
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


def test_register_and_login_basic_user(reset: None) -> None:
    """A newly registered user can log in and get their own /me info."""
    basic = UserManager.create(name="search_test_basic")
    assert basic.email is not None
    assert basic.role.value == "basic"

    resp = requests.get(
        f"{API_SERVER_URL}/me",
        headers=basic.headers,
        cookies=basic.cookies,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "basic"


def test_admin_can_list_users(admin_user: DATestUser) -> None:
    """Admin GET /manage/users/accepted returns paginated user list."""
    resp = requests.get(
        f"{API_SERVER_URL}/manage/users/accepted",
        headers=admin_user.headers,
        cookies=admin_user.cookies,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "items" in data
    assert "total_items" in data
    assert data["total_items"] >= 1


def test_basic_user_cannot_list_users(reset: None) -> None:
    """Basic user is forbidden from GET /manage/users/accepted."""
    basic = UserManager.create(name="nonadmin")
    resp = requests.get(
        f"{API_SERVER_URL}/manage/users/accepted",
        headers=basic.headers,
        cookies=basic.cookies,
    )
    assert resp.status_code == 403
