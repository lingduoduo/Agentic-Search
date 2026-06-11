"""Unit tests for src/servers/retrieval/app.py."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.internal.servers.app import (
    create_base_app,
    create_search_app,
    format_document,
)


# ---------------------------------------------------------------------------
# format_document
# ---------------------------------------------------------------------------


class TestFormatDocument:
    def test_with_title_and_content(self):
        result = format_document("My Title", "Some content")
        assert result == {
            "document": {"title": "My Title", "contents": '"My Title"\nSome content'}
        }

    def test_none_title_uses_default(self):
        result = format_document(None, "content")
        assert "No title." in result["document"]["contents"]
        assert "content" in result["document"]["contents"]

    def test_none_content_uses_default(self):
        result = format_document("Title", None)
        assert "No snippet available." in result["document"]["contents"]
        assert "Title" in result["document"]["contents"]

    def test_both_none_uses_both_defaults(self):
        result = format_document(None, None)
        assert "No title." in result["document"]["contents"]
        assert "No snippet available." in result["document"]["contents"]

    def test_returns_nested_document_structure(self):
        result = format_document("T", "C")
        assert "document" in result
        assert "contents" in result["document"]
        assert "title" in result["document"]

    def test_empty_string_title_uses_default(self):
        # Empty string is falsy → triggers default
        result = format_document("", "content")
        assert "No title." in result["document"]["contents"]

    def test_url_is_included_when_provided(self):
        result = format_document("T", "C", url="https://example.com")
        assert result["document"]["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# create_base_app  — /health endpoint
# ---------------------------------------------------------------------------


class TestCreateBaseApp:
    def setup_method(self):
        self.client = TestClient(create_base_app("Test App"))

    def test_health_returns_200(self):
        assert self.client.get("/health").status_code == 200

    def test_health_returns_ok_status(self):
        assert self.client.get("/health").json() == {"status": "ok"}

    def test_app_title_set_correctly(self):
        app = create_base_app("My Custom Title")
        assert app.title == "My Custom Title"

    def test_health_is_get_only(self):
        response = self.client.post("/health")
        assert response.status_code == 405  # Method Not Allowed


# ---------------------------------------------------------------------------
# create_search_app  — /retrieve endpoint
# ---------------------------------------------------------------------------


class TestCreateSearchApp:
    def _make_client(self, return_value=None, side_effect=None):
        engine = MagicMock()
        if side_effect is not None:
            engine.batch_search.side_effect = side_effect
        else:
            engine.batch_search.return_value = return_value
        app = create_search_app("Search App", engine)
        return TestClient(app), engine

    def test_retrieve_returns_200(self):
        client, _ = self._make_client(return_value=[["doc1"]])
        assert client.post("/retrieve", json={"queries": ["q"]}).status_code == 200

    def test_retrieve_returns_engine_results(self):
        client, _ = self._make_client(return_value=[["doc1", "doc2"]])
        data = client.post("/retrieve", json={"queries": ["q"]}).json()
        assert data["result"] == [["doc1", "doc2"]]

    def test_retrieve_calls_batch_search_with_queries(self):
        client, engine = self._make_client(return_value=[[]])
        client.post("/retrieve", json={"queries": ["hello", "world"]})
        engine.batch_search.assert_called_once_with(["hello", "world"])

    def test_retrieve_engine_exception_returns_500(self):
        client, _ = self._make_client(side_effect=RuntimeError("boom"))
        assert client.post("/retrieve", json={"queries": ["q"]}).status_code == 500

    def test_retrieve_missing_queries_field_returns_422(self):
        client, _ = self._make_client(return_value=[[]])
        assert client.post("/retrieve", json={}).status_code == 422

    def test_retrieve_empty_queries_list_returns_422(self):
        client, _ = self._make_client(return_value=[[]])
        # Field has min_length=1 constraint
        assert client.post("/retrieve", json={"queries": []}).status_code == 422

    def test_health_endpoint_still_works(self):
        client, _ = self._make_client(return_value=[[]])
        assert client.get("/health").json() == {"status": "ok"}

    def test_fetch_endpoint_is_available_when_engine_supports_fetch(self):
        engine = MagicMock()
        engine.batch_search.return_value = [[]]
        engine.fetch_urls.return_value = ["page1"]
        client = TestClient(create_search_app("Search App", engine))

        response = client.post("/fetch", json={"urls": ["https://example.com"]})

        assert response.status_code == 200
        assert response.json()["result"] == ["page1"]
        engine.fetch_urls.assert_called_once_with(["https://example.com"])


def test_google_search_server_import_does_not_require_google_client():
    from src.internal.servers.web_search import google as google_search_server

    assert google_search_server.DEFAULT_TOPK >= 1


def test_local_retrieval_server_applies_acl_filters(monkeypatch):
    from src.internal.servers.retrieval import retrieval_server

    class FakeRetriever:
        config = type("Config", (), {"topk": 5})()

        def retrieve(self, queries, topk=None):
            del queries, topk
            return [
                [
                    {
                        "document": {
                            "title": "Public",
                            "contents": '"Public"\nAllowed',
                            "metadata": {"acl": ["public"]},
                        },
                        "score": 1.0,
                    },
                    {
                        "document": {
                            "title": "Private",
                            "contents": '"Private"\nHidden',
                            "metadata": {"acl": ["group:ops"]},
                        },
                        "score": 0.5,
                    },
                ]
            ]

    monkeypatch.setattr(
        retrieval_server,
        "build_retriever",
        lambda config: FakeRetriever(),
    )

    client = TestClient(retrieval_server.create_app(FakeRetriever.config))
    response = client.post(
        "/retrieve",
        json={
            "queries": ["policy"],
            "return_scores": True,
            "filters": {"access_acl": ["public"]},
        },
    )

    assert response.status_code == 200
    row = response.json()["result"][0]
    assert [item["document"]["title"] for item in row] == ["Public"]
