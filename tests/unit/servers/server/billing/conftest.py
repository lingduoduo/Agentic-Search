"""Shared fixtures and helpers for billing tests."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings
from src.internal.configs import AuthSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import UserRecord
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app

_ADMIN = "admin-user"
_USER = "regular-user"


def _bearer(user_id: str) -> dict:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def make_test_client(
    tmp_path, monkeypatch=None, data_dir=None
) -> tuple[TestClient, AgenticSearchStore]:
    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN, email="admin@test.local"))
    store.upsert_user(UserRecord(id=_USER, email="user@test.local"))
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))
    if monkeypatch and data_dir:
        monkeypatch.setenv("AGENTIC_SEARCH_DATA_DIR", str(data_dir))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return TestClient(app), store


def make_mock_response(json_data: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = json_data
    mock_response.raise_for_status = MagicMock()
    return mock_response


def make_mock_http_client(
    method: str = "post",
    response: MagicMock | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    mock_client = MagicMock()
    mock_method = AsyncMock(return_value=response, side_effect=side_effect)
    setattr(mock_client.return_value.__aenter__.return_value, method, mock_method)
    return mock_client
