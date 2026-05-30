"""Connector interfaces for loading documents into Agentic-Search.

The shape borrows the useful parts of ingestion-oriented connector APIs:
credentials are loaded separately from settings validation, full loads and
polling are distinct capabilities, and checkpointed connectors can return a new
cursor after streaming documents.
"""

from __future__ import annotations

import abc
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, TypeAlias, TypeVar

from .models import (
    ConnectorCheckpoint,
    ConnectorFailure,
    Document,
    HierarchyNode,
    SlimDocument,
)

SecondsSinceUnixEpoch = float

GenerateDocumentsOutput = Iterator[list[Document | HierarchyNode]]
GenerateSlimDocumentOutput = Iterator[list[SlimDocument | HierarchyNode]]

CT = TypeVar("CT", bound=ConnectorCheckpoint)


@dataclass(frozen=True)
class NormalizationResult:
    """Result of a connector-specific URL normalization attempt."""

    normalized_url: str | None
    use_default: bool = False


class BaseConnector(abc.ABC, Generic[CT]):
    """Base class for all source connectors."""

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        """Load credentials and return sanitized metadata when useful."""
        del credentials
        return None

    @staticmethod
    def parse_metadata(metadata: dict[str, Any]) -> list[str]:
        """Render simple metadata values as model-readable context lines."""

        metadata_lines: list[str] = []
        for metadata_key, metadata_value in metadata.items():
            if isinstance(metadata_value, str):
                metadata_lines.append(f"{metadata_key}: {metadata_value}")
            elif isinstance(metadata_value, list) and all(
                isinstance(val, str) for val in metadata_value
            ):
                metadata_lines.append(f"{metadata_key}: {', '.join(metadata_value)}")
            else:
                raise RuntimeError(
                    "Metadata parsing supports string values or lists of strings."
                )
        return metadata_lines

    def validate_connector_settings(self) -> None:
        """Raise if connector settings or loaded credentials are invalid."""

    def set_allow_images(self, value: bool) -> None:
        """Optional hook for connectors that can include image content."""
        del value

    @classmethod
    def normalize_url(cls, url: str) -> NormalizationResult:
        """Normalize a URL to match the connector's canonical document IDs."""
        del url
        return NormalizationResult(normalized_url=None, use_default=True)

    def build_dummy_checkpoint(self) -> CT:
        return ConnectorCheckpoint(has_more=True)  # type: ignore[return-value]


class LoadConnector(BaseConnector[CT]):
    """Connector that can load a complete source state."""

    @abc.abstractmethod
    def load_from_state(self) -> GenerateDocumentsOutput:
        raise NotImplementedError


class PollConnector(BaseConnector[CT]):
    """Connector that can poll a source by time window."""

    @abc.abstractmethod
    def poll_source(
        self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> GenerateDocumentsOutput:
        raise NotImplementedError


class SlimConnector(BaseConnector[CT]):
    """Connector that can retrieve only document identities."""

    @abc.abstractmethod
    def retrieve_all_slim_docs(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> GenerateSlimDocumentOutput:
        raise NotImplementedError


class SlimConnectorWithPermSync(BaseConnector[CT]):
    """Connector that can retrieve document identities plus permissions data."""

    @abc.abstractmethod
    def retrieve_all_slim_docs_perm_sync(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> GenerateSlimDocumentOutput:
        raise NotImplementedError


class OAuthConnector(BaseConnector[CT]):
    """Connector that can perform an OAuth authorization-code flow."""

    @dataclass(frozen=True)
    class AdditionalOauthKwargs:
        """Optional typed OAuth parameters for connector-specific flows."""

    @classmethod
    @abc.abstractmethod
    def oauth_id(cls) -> str:
        """Return the stable source/provider id used for OAuth routing."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def oauth_authorization_url(
        cls,
        base_domain: str,
        state: str,
        additional_kwargs: dict[str, str],
    ) -> str:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def oauth_code_to_token(
        cls,
        base_domain: str,
        code: str,
        additional_kwargs: dict[str, str],
    ) -> dict[str, Any]:
        raise NotImplementedError


T = TypeVar("T", bound="CredentialsProviderInterface")


class CredentialsProviderInterface(abc.ABC, Generic[T]):
    """Context-managed credentials provider for dynamic connector credentials."""

    @abc.abstractmethod
    def __enter__(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_provider_key(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_credentials(self) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_credentials(self, credential_json: dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_dynamic(self) -> bool:
        raise NotImplementedError


class StaticCredentialsProvider(
    CredentialsProviderInterface["StaticCredentialsProvider"]
):
    """Simple in-memory credentials provider."""

    def __init__(self, credentials: dict[str, Any], provider_key: str = "static"):
        self._credentials = dict(credentials)
        self._provider_key = provider_key

    def __enter__(self) -> "StaticCredentialsProvider":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    def get_provider_key(self) -> str:
        return self._provider_key

    def get_credentials(self) -> dict[str, Any]:
        return dict(self._credentials)

    def set_credentials(self, credential_json: dict[str, Any]) -> None:
        self._credentials = dict(credential_json)

    def is_dynamic(self) -> bool:
        return False


class CredentialsConnector(BaseConnector[CT]):
    """Connector that can receive a credentials provider."""

    @abc.abstractmethod
    def set_credentials_provider(
        self, credentials_provider: CredentialsProviderInterface
    ) -> None:
        raise NotImplementedError


class EventConnector(BaseConnector[CT]):
    """Connector that can produce documents from an event payload."""

    @abc.abstractmethod
    def handle_event(self, event: Any) -> GenerateDocumentsOutput:
        raise NotImplementedError


CheckpointOutput: TypeAlias = Generator[
    Document | HierarchyNode | ConnectorFailure, None, CT
]

HierarchyOutput: TypeAlias = Generator[HierarchyNode, None, None]


class CheckpointedConnector(BaseConnector[CT]):
    """Connector that streams documents and returns an updated checkpoint."""

    @abc.abstractmethod
    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: CT,
    ) -> CheckpointOutput[CT]:
        raise NotImplementedError

    def validate_checkpoint_json(self, checkpoint_json: str) -> CT:
        """Validate checkpoint JSON and return a checkpoint object.

        Connectors with custom checkpoint dataclasses can override this method
        and call their checkpoint type's ``from_json`` implementation.
        """
        return ConnectorCheckpoint.from_json(checkpoint_json)  # type: ignore[return-value]


class CheckpointedConnectorWithPermSync(CheckpointedConnector[CT]):
    """Checkpointed connector that can also stream document permissions."""

    @abc.abstractmethod
    def load_from_checkpoint_with_perm_sync(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: CT,
    ) -> CheckpointOutput[CT]:
        raise NotImplementedError


class Resolver(BaseConnector[CT]):
    """Connector that can retry failed document loads."""

    @abc.abstractmethod
    def reindex(
        self,
        errors: list[ConnectorFailure],
        include_permissions: bool = False,
    ) -> Generator[Document | ConnectorFailure | HierarchyNode, None, None]:
        raise NotImplementedError


class HierarchyConnector(BaseConnector[CT]):
    """Connector that can load folder or hierarchy nodes."""

    @abc.abstractmethod
    def load_hierarchy(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
    ) -> HierarchyOutput:
        raise NotImplementedError


_BT = TypeVar("_BT")


def batched(items: Iterable[_BT], batch_size: int) -> Iterator[list[_BT]]:
    """Yield connector outputs in stable, non-empty batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    batch: list[_BT] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
