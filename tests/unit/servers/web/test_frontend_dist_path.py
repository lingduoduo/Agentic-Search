"""The built frontend bundle is found where `npm run build` actually writes it.

`_frontend_dist_path` derives the repository root by counting parents up from
`src/internal/servers/web/app.py`. Miscount by one and it silently looks for
`src/web/dist`, which nothing ever creates: the function then always returns
None and the backend serves the inline fallback shell instead of the built app,
with no error anywhere.

These tests fabricate the directory tree rather than relying on a real build,
because `web/dist` is gitignored and absent in CI.
"""

from __future__ import annotations

from pathlib import Path

from src.internal.servers.web import app as web_app


def _fake_checkout(tmp_path: Path, *, with_bundle: bool = True) -> Path:
    """Build a tree shaped like this repository; return the fake app.py path."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    module = tmp_path / "src" / "internal" / "servers" / "web" / "app.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fake\n")
    if with_bundle:
        dist = tmp_path / "web" / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html>")
    return module


def test_the_bundle_is_found_at_the_repository_root(tmp_path, monkeypatch):
    module = _fake_checkout(tmp_path)
    monkeypatch.setattr(web_app, "__file__", str(module))

    assert web_app._frontend_dist_path() == tmp_path / "web" / "dist"


def test_no_bundle_built_returns_none(tmp_path, monkeypatch):
    """The control: absent a build, the caller still falls back to the shell."""
    module = _fake_checkout(tmp_path, with_bundle=False)
    monkeypatch.setattr(web_app, "__file__", str(module))

    assert web_app._frontend_dist_path() is None


def test_an_incomplete_bundle_is_not_served(tmp_path, monkeypatch):
    """index.html without assets/ is a broken build, not a servable bundle."""
    module = _fake_checkout(tmp_path, with_bundle=False)
    dist = tmp_path / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(web_app, "__file__", str(module))

    assert web_app._frontend_dist_path() is None
