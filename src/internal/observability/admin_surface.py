"""Admin and observability summary helpers."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.db import IndexAttemptRecord
from src.internal.tools.built_in_tools import CITEABLE_TOOLS_NAMES
from src.internal.tools.built_in_tools import STOPPING_TOOLS_NAMES
from src.internal.tools.built_in_tools import TOOL_NAME_TO_CLASS

_BUILT_IN_TOOL_COUNT = len(
    CITEABLE_TOOLS_NAMES | STOPPING_TOOLS_NAMES | set(TOOL_NAME_TO_CLASS)
)


class AdminSurfaceMetric(BaseModel):
    label: str
    value: str
    detail: str


class AdminSurfaceCard(BaseModel):
    key: str
    title: str
    status: str
    tone: str = Field(pattern="^(good|watch|neutral)$")
    description: str
    items: list[str]


class AdminSurfaceSummary(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    health_label: str = "Operational readiness"
    health_score: int
    metrics: list[AdminSurfaceMetric]
    sections: list[AdminSurfaceCard]


def build_admin_surface_summary(
    store: AgenticSearchStore,
    settings: AppSettings,
) -> AdminSurfaceSummary:
    """Build one compact admin/observability snapshot from local state."""

    connectors = store.list_connectors()
    enabled_connectors = [connector for connector in connectors if connector.enabled]
    documents = store.list_documents()
    attempts = store.list_index_attempts()
    attempt_statuses = Counter(attempt.status for attempt in attempts)
    users = store.list_users()
    groups = store.list_groups()
    hooks = store.list_hooks()
    scim_tokens = store.list_scim_tokens()
    scim_user_mappings = sum(
        1 for user in users if store.get_scim_user_mapping(user.id)
    )
    scim_group_mappings = sum(
        1 for group in groups if store.get_scim_group_mapping(group.id)
    )

    failed_attempts = attempt_statuses.get("failed", 0)
    in_progress_attempts = attempt_statuses.get("in_progress", 0)
    attempted_connector_ids = {attempt.connector_id for attempt in attempts}
    stale_connectors = [
        connector
        for connector in enabled_connectors
        if connector.id not in attempted_connector_ids
    ]
    watch_count = failed_attempts + len(stale_connectors)
    health_score = max(0, min(100, 100 - watch_count * 8))

    source_counts = Counter(connector.source for connector in enabled_connectors)
    top_sources = ", ".join(
        f"{source}:{count}" for source, count in source_counts.most_common(3)
    )
    latest_attempt = attempts[0] if attempts else None
    active_hooks = sum(1 for hook in hooks if hook.is_active)

    return AdminSurfaceSummary(
        health_score=health_score,
        metrics=[
            AdminSurfaceMetric(
                label="Connectors",
                value=str(len(enabled_connectors)),
                detail=f"{len(connectors)} configured",
            ),
            AdminSurfaceMetric(
                label="Indexed docs",
                value=str(len(documents)),
                detail=_index_attempt_detail(attempt_statuses),
            ),
            AdminSurfaceMetric(
                label="Users/groups",
                value=f"{len(users)}/{len(groups)}",
                detail=f"{scim_user_mappings + scim_group_mappings} SCIM mapped",
            ),
            AdminSurfaceMetric(
                label="Tools/actions",
                value=str(_BUILT_IN_TOOL_COUNT),
                detail=f"{active_hooks} active hooks",
            ),
        ],
        sections=[
            AdminSurfaceCard(
                key="connectors",
                title="Connector management",
                status="Healthy" if not stale_connectors else "Needs sync",
                tone="good" if not stale_connectors else "watch",
                description="Source syncs, credentials, crawl windows, and permission sync.",
                items=[
                    top_sources or "No enabled connector sources",
                    f"{len(stale_connectors)} enabled connectors without attempts",
                ],
            ),
            AdminSurfaceCard(
                key="indexing",
                title="Indexing status",
                status="Watching" if failed_attempts else "Ready",
                tone="watch" if failed_attempts else "good",
                description="Fetch, parse, chunk, enrich, embed, and search-index pipeline.",
                items=[
                    f"{attempt_statuses.get('success', 0)} successful attempts",
                    f"{failed_attempts} failed, {in_progress_attempts} in progress",
                    _latest_attempt_item(latest_attempt),
                ],
            ),
            AdminSurfaceCard(
                key="access",
                title="Users and groups",
                status="Synced" if scim_tokens else "Local",
                tone="good" if scim_tokens else "neutral",
                description="Internal groups, external group mappings, and document ACLs.",
                items=[
                    f"{len(users)} users, {len(groups)} groups",
                    f"{scim_user_mappings} user mappings, {scim_group_mappings} group mappings",
                ],
            ),
            AdminSurfaceCard(
                key="auth",
                title="Auth controls",
                status="Enforced" if settings.auth.super_users else "Dev",
                tone="good" if settings.auth.super_users else "watch",
                description="SSO, API keys, tenant gates, role policies, and token limits.",
                items=[
                    f"{len(settings.auth.super_users)} super users",
                    "Super API key configured"
                    if settings.auth.super_api_key
                    else "No super API key",
                ],
            ),
            AdminSurfaceCard(
                key="models",
                title="Model settings",
                status="Ready",
                tone="neutral",
                description="Primary LLM, reasoning model, reranker, and embedding settings.",
                items=[
                    f"{settings.llm.model_provider}/{settings.llm.model_name}",
                    f"{settings.llm.max_input_tokens} max input tokens",
                ],
            ),
            AdminSurfaceCard(
                key="tools",
                title="Tools and actions",
                status="Governed" if active_hooks else "Ready",
                tone="good" if active_hooks else "neutral",
                description="Custom actions, OpenAPI tools, MCP integrations, and policies.",
                items=[
                    f"{len(CITEABLE_TOOLS_NAMES)} citeable tools",
                    f"{active_hooks}/{len(hooks)} hooks active",
                ],
            ),
            AdminSurfaceCard(
                key="analytics",
                title="Analytics",
                status="Live",
                tone="good",
                description="Sessions, citations, answer quality, latency, and usage trends.",
                items=[
                    "Session analytics endpoint enabled",
                    "User activity endpoint enabled",
                ],
            ),
            AdminSurfaceCard(
                key="enterprise",
                title="Enterprise controls",
                status="Active"
                if settings.license_enforcement_enabled
                else "Available",
                tone="good" if settings.license_enforcement_enabled else "neutral",
                description="Licensing, tenant isolation, audit hooks, and data controls.",
                items=[
                    "License enforcement on"
                    if settings.license_enforcement_enabled
                    else "License enforcement off",
                    "Cloud data plane configured"
                    if settings.cloud_data_plane_url
                    else "Local data plane",
                ],
            ),
        ],
    )


def _index_attempt_detail(statuses: Counter[str]) -> str:
    if not statuses:
        return "no index attempts"
    return ", ".join(f"{status}:{count}" for status, count in sorted(statuses.items()))


def _latest_attempt_item(attempt: IndexAttemptRecord | None) -> str:
    if attempt is None:
        return "No indexing attempts yet"
    connector_id = attempt.connector_id or "unassigned"
    return f"Latest {connector_id}: {attempt.status}"
