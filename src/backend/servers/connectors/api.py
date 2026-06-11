"""Connector management REST API.

Provides CRUD for ConnectorConfig records and a run endpoint that enqueues
an IndexAttemptRecord for the background worker.

Endpoints (prefix /admin/connectors):

  GET    /                        — list all connectors with latest attempt
  POST   /                        — create a connector
  GET    /{id}                    — connector detail + recent attempts
  PATCH  /{id}                    — update name / config / enabled flag
  DELETE /{id}                    — delete connector (and its documents)
  POST   /{id}/run                — enqueue a sync attempt

  POST   /ingest                  — push documents from an external system
                                    (no admin auth required; any caller can push)
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.db import AgenticSearchStore
from src.backend.db.models import ConnectorConfig, IndexAttemptStatus, StoredDocument
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS_RETURNED = 20


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ConnectorCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    source: str = Field(
        ...,
        description=(
            "DocumentSource value, e.g. 'file', 'web', 'rss', 'search', "
            "'github', 'slack' …"
        ),
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Connector-specific settings. "
            "file: {'paths': ['/data/docs']}. "
            "web:  {'urls': ['https://example.com'], 'max_depth': 2}. "
            "rss:  {'feed_url': 'https://example.com/feed.xml'}. "
            "search: {'queries': ['AI news'], 'provider': 'serpapi'}."
        ),
    )
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConnectorUpdateRequest(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class IndexAttemptView(BaseModel):
    id: str
    status: IndexAttemptStatus
    total_documents: int
    total_chunks: int
    error: str | None
    created_at: str | None
    updated_at: str | None


class ConnectorView(BaseModel):
    id: str
    name: str
    source: str
    config: dict[str, Any]
    enabled: bool
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None
    last_attempt: IndexAttemptView | None = None


class ConnectorDetailView(ConnectorView):
    attempts: list[IndexAttemptView] = Field(default_factory=list)
    document_count: int = 0


class IngestDocument(BaseModel):
    id: str | None = None
    title: str
    contents: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    connector_id: str | None = Field(
        None,
        description="Associate documents with an existing connector, or omit to use a shared ingestion connector.",
    )
    documents: list[IngestDocument] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    connector_id: str


class RunResponse(BaseModel):
    attempt_id: str
    connector_id: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attempt_view(attempt) -> IndexAttemptView:
    return IndexAttemptView(
        id=attempt.id,
        status=attempt.status,
        total_documents=attempt.total_documents,
        total_chunks=attempt.total_chunks,
        error=attempt.error,
        created_at=attempt.created_at,
        updated_at=attempt.updated_at,
    )


def _connector_view(
    connector: ConnectorConfig,
    store: AgenticSearchStore,
    *,
    include_attempts: bool = False,
) -> ConnectorView | ConnectorDetailView:
    attempts = store.list_index_attempts(connector_id=connector.id)[
        :_MAX_ATTEMPTS_RETURNED
    ]
    last = _attempt_view(attempts[0]) if attempts else None

    base = dict(
        id=connector.id,
        name=connector.name,
        source=connector.source,
        config=connector.config,
        enabled=connector.enabled,
        metadata=connector.metadata,
        created_at=connector.created_at,
        updated_at=connector.updated_at,
        last_attempt=last,
    )
    if include_attempts:
        doc_count = len(store.list_documents(connector_id=connector.id, limit=None))
        return ConnectorDetailView(
            **base,
            attempts=[_attempt_view(a) for a in attempts],
            document_count=doc_count,
        )
    return ConnectorView(**base)


def _get_or_404(store: AgenticSearchStore, connector_id: str) -> ConnectorConfig:
    connector = store.get_connector(connector_id)
    if connector is None:
        raise HTTPException(
            status_code=404, detail=f"Connector {connector_id!r} not found."
        )
    return connector


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_connectors_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with connector CRUD + run + ingest endpoints."""

    router = APIRouter(prefix="/admin/connectors", tags=["connectors"])
    _require_admin = make_require_admin(app_settings)

    @router.get("")
    def list_connectors(
        enabled: bool | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[ConnectorView]:
        """Return all connectors with their latest index attempt."""
        connectors = store.list_connectors(enabled=enabled)
        return [_connector_view(c, store) for c in connectors]  # type: ignore[return-value]

    @router.post("", status_code=201)
    def create_connector(
        req: ConnectorCreateRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> ConnectorView:
        """Create a new connector configuration."""
        existing = [c for c in store.list_connectors() if c.name == req.name]
        if existing:
            raise HTTPException(
                status_code=409, detail=f"Connector named {req.name!r} already exists."
            )
        connector = store.upsert_connector(
            ConnectorConfig(
                id=f"connector_{uuid4().hex[:12]}",
                name=req.name,
                source=req.source,
                config=req.config,
                enabled=req.enabled,
                metadata=req.metadata,
            )
        )
        logger.info("Connector created: %s (%s)", connector.name, connector.id)
        return _connector_view(connector, store)  # type: ignore[return-value]

    @router.get("/{connector_id}")
    def get_connector(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> ConnectorDetailView:
        """Return full connector detail including recent index attempts."""
        connector = _get_or_404(store, connector_id)
        return _connector_view(connector, store, include_attempts=True)  # type: ignore[return-value]

    @router.patch("/{connector_id}")
    def update_connector(
        connector_id: str,
        req: ConnectorUpdateRequest,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> ConnectorView:
        """Update name, config, enabled state, or metadata of a connector."""
        connector = _get_or_404(store, connector_id)
        updated = ConnectorConfig(
            id=connector.id,
            name=req.name if req.name is not None else connector.name,
            source=connector.source,
            config=req.config if req.config is not None else connector.config,
            enabled=req.enabled if req.enabled is not None else connector.enabled,
            metadata=req.metadata if req.metadata is not None else connector.metadata,
        )
        saved = store.upsert_connector(updated)
        logger.info("Connector updated: %s", connector_id)
        return _connector_view(saved, store)  # type: ignore[return-value]

    @router.delete("/{connector_id}", status_code=204)
    def delete_connector(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> None:
        """Delete a connector and all its stored documents and index attempts."""
        _get_or_404(store, connector_id)
        store.delete_connector(connector_id)
        logger.info("Connector deleted: %s", connector_id)

    @router.post("/{connector_id}/run", status_code=202)
    def run_connector(
        connector_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> RunResponse:
        """Enqueue a sync attempt for the connector.

        The background worker picks up attempts with status ``not_started``
        and drives the connector's load pipeline.
        """
        _get_or_404(store, connector_id)
        attempt = store.create_index_attempt(connector_id=connector_id)
        logger.info("Run enqueued: connector=%s attempt=%s", connector_id, attempt.id)
        return RunResponse(
            attempt_id=attempt.id,
            connector_id=connector_id,
            message=f"Sync attempt {attempt.id} queued.",
        )

    # --- Ingest endpoint (no admin auth) ---

    @router.post("/ingest", status_code=200)
    def ingest_documents(req: IngestRequest) -> IngestResponse:
        """Push documents from an external system directly into the store.

        If *connector_id* is omitted, a shared ``ingestion_api`` connector is
        created on first use and reused on subsequent calls.
        """
        connector_id = req.connector_id
        if connector_id is None:
            _INGEST_CONNECTOR_NAME = "_ingestion_api"
            existing = [
                c for c in store.list_connectors() if c.name == _INGEST_CONNECTOR_NAME
            ]
            if existing:
                connector_id = existing[0].id
            else:
                c = store.upsert_connector(
                    ConnectorConfig(
                        id=f"connector_{uuid4().hex[:12]}",
                        name=_INGEST_CONNECTOR_NAME,
                        source="ingestion_api",
                    )
                )
                connector_id = c.id
        else:
            _get_or_404(store, connector_id)

        ingested = 0
        for doc in req.documents:
            doc_id = doc.id or f"ingest:{uuid4().hex}"
            store.upsert_document(
                StoredDocument(
                    id=doc_id,
                    title=doc.title,
                    contents=doc.contents,
                    url=doc.url,
                    connector_id=connector_id,
                    metadata={**doc.metadata, "source": "ingestion_api"},
                )
            )
            ingested += 1

        return IngestResponse(ingested=ingested, connector_id=connector_id)

    return router


__all__ = ["create_connectors_router"]
