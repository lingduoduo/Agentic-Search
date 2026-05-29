from __future__ import annotations

from src.feature_flags import EnvFeatureFlagProvider
from src.feature_flags import PostHogFeatureFlagProvider
from src.feature_flags import StaticFeatureFlagProvider
from src.feature_flags import get_feature_flag_provider
from src.feature_flags import is_feature_enabled
from src.feature_flags import normalize_flag_key


def test_env_feature_flags_use_agentic_and_fallback_prefixes():
    provider = EnvFeatureFlagProvider(
        {
            "AGENTIC_SEARCH_FEATURE_NEW_SEARCH": "true",
            "FEATURE_LEGACY_MODE": "1",
        }
    )

    assert provider.feature_enabled("new-search") is True
    assert provider.feature_enabled("legacy mode") is True
    assert provider.feature_enabled("missing") is False


def test_static_feature_flags_normalize_keys():
    provider = StaticFeatureFlagProvider({"NEW_SEARCH": True})

    assert normalize_flag_key("new.search") == "NEW_SEARCH"
    assert provider.feature_enabled("new-search") is True


def test_posthog_provider_uses_injected_client_and_default_on_error():
    class FakePostHog:
        def __init__(self):
            self.calls = []

        def set(self, distinct_id, properties):
            self.calls.append(("set", distinct_id, properties))

        def feature_enabled(self, flag_key, distinct_id, person_properties=None):
            self.calls.append((flag_key, distinct_id, person_properties))
            return flag_key == "enabled-flag"

    client = FakePostHog()
    provider = PostHogFeatureFlagProvider(client=client)

    assert provider.feature_enabled(
        "enabled-flag", user_id="u1", user_properties={"tier": "business"}
    )
    assert not provider.feature_enabled("disabled-flag", user_id="u1")
    assert client.calls[0] == ("set", "u1", {"tier": "business"})


def test_default_provider_prefers_env_over_posthog():
    class FakePostHog:
        def feature_enabled(self, flag_key, distinct_id, person_properties=None):
            del flag_key, distinct_id, person_properties
            return True

    provider = get_feature_flag_provider(
        env={"AGENTIC_SEARCH_FEATURE_REMOTE_ONLY": "false"},
        posthog_client=FakePostHog(),
    )

    assert is_feature_enabled("missing_remote", user_id="u1", provider=provider) is True
    assert is_feature_enabled("remote_only", user_id="u1", provider=provider) is False
    assert (
        is_feature_enabled(
            "REMOTE_ONLY",
            provider=EnvFeatureFlagProvider(
                {"AGENTIC_SEARCH_FEATURE_REMOTE_ONLY": "false"}
            ),
        )
        is False
    )
