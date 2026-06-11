from __future__ import annotations

import pytest

from src.internal.configs import Tier
from src.internal.configs import get_env_bool
from src.internal.configs import is_license_enforcement_exempt
from src.internal.configs import is_path_allowed_for_gated_tenant
from src.internal.configs import is_path_allowed_for_tier
from src.internal.configs import load_app_settings
from src.internal.configs import load_permission_sync_settings
from src.internal.configs import required_tier_for_path
from src.internal.servers.web.app import SearchExperienceSettings


def test_load_app_settings_reads_typed_environment():
    settings = load_app_settings(
        {
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
            "AGENTIC_SEARCH_WEB_TOP_K": "7",
            "AGENTIC_SEARCH_AUTH_SECRET": "secret",
            "AGENTIC_SEARCH_SUPER_USERS": '["admin@example.test"]',
            "POSTHOG_DEBUG_LOGS_ENABLED": "true",
        }
    )

    assert settings.services.retrieval_url == "http://search.test/retrieve"
    assert settings.services.web_top_k == 7
    assert settings.auth.secret == "secret"
    assert settings.auth.super_users == ("admin@example.test",)
    assert settings.telemetry.posthog_debug_logs_enabled is True


def test_bool_env_parser_rejects_ambiguous_values():
    assert get_env_bool({"FLAG": "yes"}, "FLAG") is True
    assert get_env_bool({"FLAG": "0"}, "FLAG", True) is False

    with pytest.raises(ValueError, match="FLAG"):
        get_env_bool({"FLAG": "sometimes"}, "FLAG")


def test_permission_sync_settings_keep_connector_specific_defaults():
    settings = load_permission_sync_settings(
        {
            "DEFAULT_PERMISSION_DOC_SYNC_FREQUENCY": "42",
            "CONFLUENCE_PERMISSION_DOC_SYNC_FREQUENCY": "1800",
            "GITHUB_PERMISSION_GROUP_SYNC_FREQUENCY": "60",
            "CONFLUENCE_ANONYMOUS_ACCESS_IS_PUBLIC": "true",
        }
    )

    assert settings.doc_sync_frequency("unknown") == 42
    assert settings.doc_sync_frequency("confluence") == 1800
    assert settings.group_sync_frequency("github") == 60
    assert settings.anonymous_access_is_public("confluence") is True


def test_license_tier_gating_uses_longest_prefix_and_allowlist():
    assert is_license_enforcement_exempt("/health/live")
    assert required_tier_for_path("/admin/hooks/outbound") == Tier.ENTERPRISE
    assert is_path_allowed_for_tier("/admin/api-key", Tier.BUSINESS)
    assert not is_path_allowed_for_tier("/analytics/team", Tier.BUSINESS)
    assert is_path_allowed_for_tier("/analytics/team", Tier.ENTERPRISE)


def test_multi_tenant_gating_allowlist_is_prefix_based():
    assert is_path_allowed_for_gated_tenant("/tenants/create-subscription-session")
    assert is_path_allowed_for_gated_tenant("/assets/app.css")
    assert not is_path_allowed_for_gated_tenant("/api/agent")


def test_web_settings_can_be_built_from_app_settings():
    app_settings = load_app_settings(
        {
            "AGENTIC_SEARCH_RETRIEVAL_URL": "http://search.test/retrieve",
            "AGENTIC_SEARCH_WEB_TOP_K": "9",
            "AGENTIC_SEARCH_WEB_DB_PATH": "/tmp/search.sqlite3",
        }
    )

    web_settings = SearchExperienceSettings.from_app_settings(app_settings)

    assert web_settings.search_url == "http://search.test/retrieve"
    assert web_settings.top_k == 9
    assert web_settings.db_path == "/tmp/search.sqlite3"
