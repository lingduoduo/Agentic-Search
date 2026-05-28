"""Unit tests for the web-search provider admin API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.servers.web_search import WebSearchProviderStore
from src.servers.web_search import create_app


def _client() -> TestClient:
    return TestClient(create_app(WebSearchProviderStore()))


def test_search_provider_lifecycle_masks_key_and_activates_provider():
    client = _client()

    response = client.post(
        "/admin/web-search/search-providers",
        json={
            "name": "SerpAPI",
            "provider_type": "serpapi",
            "api_key": "serp-secret-key",
            "api_key_changed": True,
            "activate": True,
            "config": {"page_size": 3},
        },
    )

    assert response.status_code == 200
    provider = response.json()
    assert provider["id"] == 1
    assert provider["is_active"] is True
    assert provider["masked_api_key"] == "serp...-key"

    list_response = client.get("/admin/web-search/search-providers")
    assert list_response.json()[0]["name"] == "SerpAPI"

    deactivate_response = client.post("/admin/web-search/search-providers/1/deactivate")
    assert deactivate_response.json() == {"status": "ok"}
    assert (
        client.get("/admin/web-search/search-providers").json()[0]["is_active"] is False
    )

    delete_response = client.delete("/admin/web-search/search-providers/1")
    assert delete_response.status_code == 204
    assert client.get("/admin/web-search/search-providers").json() == []


def test_search_provider_requires_unique_names():
    client = _client()
    payload = {
        "name": "Local retrieval",
        "provider_type": "retrieval",
        "config": {"search_url": "http://localhost:8000/retrieve"},
    }

    response = client.post("/admin/web-search/search-providers", json=payload)
    assert response.status_code == 200
    response = client.post("/admin/web-search/search-providers", json=payload)

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_provider_test_can_use_stored_key_without_live_network_call():
    client = _client()
    client.post(
        "/admin/web-search/search-providers",
        json={
            "name": "Google",
            "provider_type": "google",
            "api_key": "google-secret-key",
            "api_key_changed": True,
            "activate": True,
            "config": {"cse_id": "cx"},
        },
    )

    response = client.post(
        "/admin/web-search/search-providers/test",
        json={
            "provider_type": "google",
            "use_stored_key": True,
            "config": {"cse_id": "cx"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_google_provider_test_requires_key_and_cse_id():
    client = _client()

    missing_key = client.post(
        "/admin/web-search/search-providers/test",
        json={"provider_type": "google", "config": {"cse_id": "cx"}},
    )
    assert missing_key.status_code == 400
    assert "API key" in missing_key.json()["detail"]

    missing_cse = client.post(
        "/admin/web-search/search-providers/test",
        json={"provider_type": "google", "api_key": "key", "config": {}},
    )
    assert missing_cse.status_code == 400
    assert "cse_id" in missing_cse.json()["detail"]


def test_content_provider_reset_deactivates_active_provider():
    client = _client()
    response = client.post(
        "/admin/web-search/content-providers",
        json={"name": "Direct", "provider_type": "direct", "activate": True},
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is True

    assert client.post("/admin/web-search/content-providers/reset-default").json() == {
        "status": "ok"
    }
    providers = client.get("/admin/web-search/content-providers").json()
    assert providers[0]["is_active"] is False


def test_retrieval_provider_requires_search_url():
    client = _client()

    response = client.post(
        "/admin/web-search/search-providers",
        json={"name": "Retrieval", "provider_type": "retrieval"},
    )

    assert response.status_code == 400
    assert "search_url" in response.json()["detail"]
