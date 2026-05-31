"""Tests for formerly-stub functions in process_message.py and save_chat.py."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.backend.chat.process_message import (
    AppError,
    AppErrorCode,
    _provider_token_windows,
    check_llm_cost_limit_for_provider,
    verify_user_files,
)
from src.backend.chat.save_chat import (
    add_search_docs_to_chat_message,
    add_search_docs_to_tool_call,
)


# ---------------------------------------------------------------------------
# verify_user_files
# ---------------------------------------------------------------------------


def _make_file(id_: str):
    @dataclass
    class FD:
        id: str

    return FD(id=id_)


def test_verify_user_files_empty_list_passes():
    verify_user_files(user_files=[], user_id="u1", db_session=None)


def test_verify_user_files_no_id_attr_passes():
    verify_user_files(user_files=[object()], user_id="u1", db_session=None)


def test_verify_user_files_raises_on_missing_file():
    with pytest.raises(AppError) as exc_info:
        verify_user_files(
            user_files=[_make_file("missing-file-123")],
            user_id="u1",
            db_session=None,
        )
    assert exc_info.value.error_code is AppErrorCode.INSUFFICIENT_PERMISSIONS
    assert exc_info.value.status_code == 403


def test_verify_user_files_passes_when_file_exists():
    """If read_file_record doesn't raise, no error should be raised."""

    class _StoreWithFile:
        def read_file_record(self, fid):
            return {"id": fid}

    with patch(
        "src.backend.chat.process_message.get_default_file_store",
        return_value=_StoreWithFile(),
    ):
        verify_user_files(
            user_files=[_make_file("existing-file-456")],
            user_id="u1",
            db_session=None,
        )


# ---------------------------------------------------------------------------
# check_llm_cost_limit_for_provider
# ---------------------------------------------------------------------------


def _clear_windows():
    _provider_token_windows.clear()


def test_cost_limit_no_env_var_always_passes(monkeypatch):
    monkeypatch.delenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", raising=False)
    _clear_windows()
    for _ in range(100):
        check_llm_cost_limit_for_provider(llm_provider_api_key="sk-test")


def test_cost_limit_invalid_env_var_passes(monkeypatch):
    monkeypatch.setenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", "not-a-number")
    _clear_windows()
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-test")


def test_cost_limit_allows_up_to_limit(monkeypatch):
    monkeypatch.setenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", "3")
    _clear_windows()
    for _ in range(3):
        check_llm_cost_limit_for_provider(llm_provider_api_key="sk-abc")


def test_cost_limit_raises_on_exceed(monkeypatch):
    monkeypatch.setenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", "2")
    _clear_windows()
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-xyz")
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-xyz")
    with pytest.raises(AppError) as exc_info:
        check_llm_cost_limit_for_provider(llm_provider_api_key="sk-xyz")
    assert exc_info.value.status_code == 429


def test_cost_limit_separate_keys_independent(monkeypatch):
    monkeypatch.setenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", "1")
    _clear_windows()
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-key-A")
    # Different key — should not be affected by the first
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-key-B")


def test_cost_limit_expired_window_resets(monkeypatch):
    import time

    monkeypatch.setenv("GEN_AI_TOKEN_LIMIT_PER_MINUTE", "1")
    _clear_windows()

    # Manually pre-populate an expired timestamp
    key = "sk-expir"[:8]
    _provider_token_windows[key] = [time.monotonic() - 120]  # 2 min ago — expired

    # Should pass because the old timestamp is outside the 60s window
    check_llm_cost_limit_for_provider(llm_provider_api_key="sk-expired-key")


# ---------------------------------------------------------------------------
# add_search_docs_to_tool_call
# ---------------------------------------------------------------------------


def _make_session(info=None):
    session = MagicMock()
    session.info = info if info is not None else {}
    return session


def test_add_search_docs_to_tool_call_empty_list_noop():
    session = _make_session()
    session.get.return_value = None
    add_search_docs_to_tool_call(tool_call_id=1, search_doc_ids=[], db_session=session)
    session.get.assert_not_called()


def test_add_search_docs_to_tool_call_sets_attr_when_available():
    from src.backend.db.models import ToolCall

    obj = ToolCall(id=1)
    obj.search_doc_ids = []  # type: ignore[attr-defined]
    session = _make_session()
    session.get.return_value = obj

    add_search_docs_to_tool_call(
        tool_call_id=1, search_doc_ids=[10, 20], db_session=session
    )

    assert set(obj.search_doc_ids) == {10, 20}  # type: ignore[attr-defined]


def test_add_search_docs_to_tool_call_falls_back_to_session_info():
    session = _make_session()
    session.get.return_value = None  # not in identity map

    add_search_docs_to_tool_call(
        tool_call_id=99, search_doc_ids=[5, 6, 7], db_session=session
    )

    assert session.info["_tool_call_search_docs"][99] == [5, 6, 7]


def test_add_search_docs_to_tool_call_deduplicates_in_session_info():
    session = _make_session()
    session.get.return_value = None
    add_search_docs_to_tool_call(
        tool_call_id=99, search_doc_ids=[1, 2], db_session=session
    )
    add_search_docs_to_tool_call(
        tool_call_id=99, search_doc_ids=[2, 3], db_session=session
    )
    assert session.info["_tool_call_search_docs"][99] == [1, 2, 3]


# ---------------------------------------------------------------------------
# add_search_docs_to_chat_message
# ---------------------------------------------------------------------------


def test_add_search_docs_to_chat_message_empty_list_noop():
    session = _make_session()
    session.get.return_value = None
    add_search_docs_to_chat_message(
        chat_message_id=1, search_doc_ids=[], db_session=session
    )
    session.get.assert_not_called()


def test_add_search_docs_to_chat_message_falls_back_to_session_info():
    session = _make_session()
    session.get.return_value = None

    add_search_docs_to_chat_message(
        chat_message_id=42, search_doc_ids=[100, 200], db_session=session
    )

    assert session.info["_chat_message_search_docs"][42] == [100, 200]


def test_add_search_docs_to_chat_message_deduplicates():
    session = _make_session()
    session.get.return_value = None
    add_search_docs_to_chat_message(
        chat_message_id=42, search_doc_ids=[1, 2], db_session=session
    )
    add_search_docs_to_chat_message(
        chat_message_id=42, search_doc_ids=[2, 3], db_session=session
    )
    assert session.info["_chat_message_search_docs"][42] == [1, 2, 3]
