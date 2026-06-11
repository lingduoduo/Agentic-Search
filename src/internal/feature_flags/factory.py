"""Feature flag providers for Agentic Search."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.internal.configs import get_env_bool
from src.internal.configs import load_app_settings


class FeatureFlagProvider(Protocol):
    """Minimal provider protocol used by app code and tests."""

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        """Return whether *flag_key* is enabled for the caller."""


@dataclass(frozen=True)
class NoOpFeatureFlagProvider:
    """Provider that always returns the caller's default."""

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        del flag_key, user_id, user_properties
        return default


@dataclass(frozen=True)
class StaticFeatureFlagProvider:
    """Provider backed by an explicit mapping."""

    flags: Mapping[str, bool] = field(default_factory=dict, hash=False, compare=False)

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        del user_id, user_properties
        return bool(self.flags.get(normalize_flag_key(flag_key), default))


@dataclass(frozen=True)
class EnvFeatureFlagProvider:
    """Provider that reads feature flags from environment variables.

    Both ``AGENTIC_SEARCH_FEATURE_FOO`` and ``FEATURE_FOO`` are supported. Flag
    names are normalized to uppercase snake case.
    """

    env: Mapping[str, str] = field(
        default_factory=lambda: os.environ, hash=False, compare=False
    )
    prefix: str = "AGENTIC_SEARCH_FEATURE_"
    fallback_prefix: str = "FEATURE_"

    def has_flag(self, flag_key: str) -> bool:
        normalized = normalize_flag_key(flag_key)
        return any(
            env_key in self.env
            for env_key in (
                f"{self.prefix}{normalized}",
                f"{self.fallback_prefix}{normalized}",
            )
        )

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        del user_id, user_properties
        normalized = normalize_flag_key(flag_key)
        for env_key in (
            f"{self.prefix}{normalized}",
            f"{self.fallback_prefix}{normalized}",
        ):
            if env_key in self.env:
                return get_env_bool(self.env, env_key, default)
        return default


@dataclass(frozen=True)
class CompositeFeatureFlagProvider:
    """Try providers in order until one returns a non-default answer."""

    providers: tuple[FeatureFlagProvider, ...]

    def feature_enabled(
        self,
        flag_key: str,
        user_id: str | None = None,
        user_properties: dict[str, Any] | None = None,
        *,
        default: bool = False,
    ) -> bool:
        for index, provider in enumerate(self.providers):
            has_flag = getattr(provider, "has_flag", None)
            if callable(has_flag):
                if not has_flag(flag_key):
                    continue
                return provider.feature_enabled(
                    flag_key,
                    user_id=user_id,
                    user_properties=user_properties,
                    default=default,
                )
            value = provider.feature_enabled(
                flag_key,
                user_id=user_id,
                user_properties=user_properties,
                default=default,
            )
            if value != default or index == len(self.providers) - 1:
                return value
        return default


def normalize_flag_key(flag_key: str) -> str:
    return (
        flag_key.strip().replace("-", "_").replace(".", "_").replace(" ", "_").upper()
    )


def get_feature_flag_provider(
    *,
    env: Mapping[str, str] | None = None,
    posthog_client: Any | None = None,
) -> FeatureFlagProvider:
    """Build the default provider chain.

    Local development works with env flags only. If a PostHog-like client is
    provided, it is consulted after env overrides.
    """

    from .posthog_provider import PostHogFeatureFlagProvider

    providers: list[FeatureFlagProvider] = [EnvFeatureFlagProvider(env or os.environ)]
    settings = load_app_settings(env)
    if posthog_client is not None or settings.telemetry.posthog_api_key:
        providers.append(PostHogFeatureFlagProvider(client=posthog_client))
    return CompositeFeatureFlagProvider(tuple(providers))


def is_feature_enabled(
    flag_key: str,
    *,
    user_id: str | None = None,
    user_properties: dict[str, Any] | None = None,
    default: bool = False,
    provider: FeatureFlagProvider | None = None,
) -> bool:
    resolved_provider = provider or get_feature_flag_provider()
    return resolved_provider.feature_enabled(
        flag_key,
        user_id=user_id,
        user_properties=user_properties,
        default=default,
    )


__all__ = [
    "CompositeFeatureFlagProvider",
    "EnvFeatureFlagProvider",
    "FeatureFlagProvider",
    "NoOpFeatureFlagProvider",
    "StaticFeatureFlagProvider",
    "get_feature_flag_provider",
    "is_feature_enabled",
    "normalize_flag_key",
]
