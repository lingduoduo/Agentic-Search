"""Shared document models for connector implementations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


class ConnectorStopSignal(Exception):
    """Raised to signal a connector run should stop cleanly."""


@dataclass(frozen=True)
class Document:
    """One source document emitted by a connector."""

    id: str
    contents: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlimDocument:
    """Minimal document identity used by lightweight sync passes."""

    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchyNode:
    """Optional parent/child node for sources with folder-like structure."""

    id: str
    name: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorFailure:
    """Structured failure emitted while a connector continues processing."""

    document_id: str | None
    message: str
    exception_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorCheckpoint:
    """Small checkpoint object for incremental connectors."""

    has_more: bool = False
    cursor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the checkpoint for persistence."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectorCheckpoint":
        """Build a checkpoint, ignoring unknown keys for forward compatibility."""
        field_names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in field_names})

    @classmethod
    def from_json(cls, checkpoint_json: str) -> "ConnectorCheckpoint":
        """Deserialize a checkpoint from a JSON object string."""
        data = json.loads(checkpoint_json)
        if not isinstance(data, dict):
            raise ValueError("Checkpoint JSON must be an object.")
        return cls.from_dict(data)
