"""DEBUG_PANELS gate: debug router mounts only when enabled."""

from __future__ import annotations

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _debug_routes(app) -> list[str]:
    return [
        r.path for r in app.routes if getattr(r, "path", "").startswith("/api/debug")
    ]


def test_debug_router_absent_by_default(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    assert _debug_routes(app) == []


def test_debug_router_present_when_enabled(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3", debug_panels=True)
    )
    assert "/api/debug/retrieval/{mode}" in _debug_routes(app)
