# tests/unit/test_cli_auth.py
from __future__ import annotations

import pytest

from src.cli._auth import resolve_token


def test_pre_baked_token_returned_as_is():
    token = resolve_token(token="my.jwt.here", user_id=None, email=None, secret=None)
    assert token == "my.jwt.here"


def test_token_takes_priority_over_user_id():
    token = resolve_token(token="pre.baked", user_id="alice", email=None, secret=None)
    assert token == "pre.baked"


def test_user_id_mints_jwt(monkeypatch):
    minted = "minted.jwt"
    monkeypatch.setattr(
        "src.cli._auth.generate_user_jwt_token",
        lambda **_: minted,
    )
    result = resolve_token(token=None, user_id="alice", email="a@corp.com", secret="s")
    assert result == minted


def test_no_token_no_user_id_raises():
    with pytest.raises(ValueError, match="--token or --user-id"):
        resolve_token(token=None, user_id=None, email=None, secret=None)
