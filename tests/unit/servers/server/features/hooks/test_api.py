"""Tests for hook API helper functions (src/servers/features/hooks/api.py).

Covers:
- _check_ssrf_safety: scheme enforcement and private-IP blocklist
- _validate_endpoint: HTTP response codes → HookValidateStatus mapping
- _raise_for_validation: HookValidateStatus → HTTPException mapping
- HookValidateStatus: string values (API contract)
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from src.internal.servers.features.hooks.api import HookValidateResponse
from src.internal.servers.features.hooks.api import HookValidateStatus
from src.internal.servers.features.hooks.api import _check_ssrf_safety
from src.internal.servers.features.hooks.api import _raise_for_validation
from src.internal.servers.features.hooks.api import _validate_endpoint

_URL = "https://example.com/hook"
_API_KEY = "secret"
_TIMEOUT = 5.0


def _mock_response(status_code: int) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    return response


# ---------------------------------------------------------------------------
# _check_ssrf_safety
# ---------------------------------------------------------------------------


class TestCheckSsrfSafety:
    def _call(self, url: str) -> None:
        _check_ssrf_safety(url)

    def test_https_is_allowed(self) -> None:
        with patch("socket.gethostbyname", return_value="93.184.216.34"):
            self._call("https://example.com/hook")  # must not raise

    def test_non_http_non_https_scheme_rejected(self) -> None:
        # Only ftp / file / etc. are rejected; http is allowed
        with pytest.raises(HTTPException) as exc_info:
            self._call("ftp://example.com/hook")
        assert exc_info.value.status_code == 400

    def test_http_scheme_is_allowed(self) -> None:
        with patch("socket.gethostbyname", return_value="93.184.216.34"):
            self._call("http://example.com/hook")  # must not raise

    @pytest.mark.parametrize(
        "ip",
        [
            pytest.param("127.0.0.1", id="loopback"),
            pytest.param("10.0.0.1", id="RFC1918-A"),
            pytest.param("172.16.0.1", id="RFC1918-B"),
            pytest.param("192.168.1.1", id="RFC1918-C"),
        ],
    )
    def test_private_ip_is_blocked(self, ip: str) -> None:
        with (
            patch("socket.gethostbyname", return_value=ip),
            pytest.raises(HTTPException) as exc_info,
        ):
            self._call("https://internal.example.com/hook")
        assert exc_info.value.status_code == 400

    def test_public_ip_is_allowed(self) -> None:
        with patch("socket.gethostbyname", return_value="93.184.216.34"):
            self._call("https://example.com/hook")  # must not raise

    def test_dns_failure_raises(self) -> None:
        import socket

        with (
            patch(
                "socket.gethostbyname",
                side_effect=socket.gaierror("name not found"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            self._call("https://no-such-host.example.com/hook")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _validate_endpoint
# ---------------------------------------------------------------------------


class TestValidateEndpoint:
    def _call(self, *, api_key: str | None = _API_KEY) -> HookValidateResponse:
        with patch("src.internal.servers.features.hooks.api._check_ssrf_safety"):
            return _validate_endpoint(
                endpoint_url=_URL,
                api_key=api_key,
                timeout_seconds=_TIMEOUT,
            )

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_2xx_returns_passed(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = (
            _mock_response(200)
        )
        assert self._call().status == HookValidateStatus.passed

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_5xx_returns_passed(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = (
            _mock_response(500)
        )
        assert self._call().status == HookValidateStatus.passed

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    @pytest.mark.parametrize("status_code", [401, 403])
    def test_401_403_returns_auth_failed(
        self, mock_client_cls: MagicMock, status_code: int
    ) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = (
            _mock_response(status_code)
        )
        result = self._call()
        assert result.status == HookValidateStatus.auth_failed
        assert str(status_code) in (result.error_message or "")

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_4xx_non_auth_returns_passed(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = (
            _mock_response(422)
        )
        assert self._call().status == HookValidateStatus.passed

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_connect_timeout_returns_timeout(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
            httpx.ConnectTimeout("timed out")
        )
        assert self._call().status == HookValidateStatus.timeout

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ReadTimeout("read timeout"),
            httpx.WriteTimeout("write timeout"),
            httpx.PoolTimeout("pool timeout"),
        ],
    )
    def test_other_timeout_returns_timeout(
        self, mock_client_cls: MagicMock, exc: httpx.TimeoutException
    ) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = exc
        assert self._call().status == HookValidateStatus.timeout

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_connect_error_returns_cannot_connect(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
            httpx.ConnectError("dns failure")
        )
        assert self._call().status == HookValidateStatus.cannot_connect

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_arbitrary_exception_returns_cannot_connect(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
            ConnectionRefusedError("refused")
        )
        assert self._call().status == HookValidateStatus.cannot_connect

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_api_key_sent_as_bearer(self, mock_client_cls: MagicMock) -> None:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = _mock_response(200)
        self._call(api_key="mykey")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer mykey"

    @patch("src.internal.servers.features.hooks.api.httpx.Client")
    def test_no_api_key_omits_auth_header(self, mock_client_cls: MagicMock) -> None:
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = _mock_response(200)
        self._call(api_key=None)
        _, kwargs = mock_post.call_args
        assert "Authorization" not in kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# _raise_for_validation
# ---------------------------------------------------------------------------


class TestRaiseForValidation:
    """_raise_for_validation always raises — callers only invoke it on failure."""

    def test_auth_failed_is_401(self) -> None:
        # Our implementation uses 401 for auth_failed (not 403)
        validation = HookValidateResponse(
            status=HookValidateStatus.auth_failed, error_message="bad key"
        )
        with pytest.raises(HTTPException) as exc_info:
            _raise_for_validation(validation)
        assert exc_info.value.status_code == 401

    def test_timeout_is_504(self) -> None:
        validation = HookValidateResponse(
            status=HookValidateStatus.timeout, error_message="timeout"
        )
        with pytest.raises(HTTPException) as exc_info:
            _raise_for_validation(validation)
        assert exc_info.value.status_code == 504

    def test_cannot_connect_is_502(self) -> None:
        validation = HookValidateResponse(
            status=HookValidateStatus.cannot_connect, error_message="refused"
        )
        with pytest.raises(HTTPException) as exc_info:
            _raise_for_validation(validation)
        assert exc_info.value.status_code == 502

    def test_error_message_passed_to_exception(self) -> None:
        validation = HookValidateResponse(
            status=HookValidateStatus.auth_failed, error_message="bad credentials"
        )
        with pytest.raises(HTTPException) as exc_info:
            _raise_for_validation(validation)
        assert "bad credentials" in exc_info.value.detail


# ---------------------------------------------------------------------------
# HookValidateStatus string values
# ---------------------------------------------------------------------------


class TestHookValidateStatusValues:
    @pytest.mark.parametrize(
        "status, expected",
        [
            (HookValidateStatus.passed, "passed"),
            (HookValidateStatus.auth_failed, "auth_failed"),
            (HookValidateStatus.timeout, "timeout"),
            (HookValidateStatus.cannot_connect, "cannot_connect"),
        ],
    )
    def test_string_values(self, status: HookValidateStatus, expected: str) -> None:
        assert status == expected
