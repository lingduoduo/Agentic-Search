"""Concrete connector implementations for common local sources."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG
from typing import Any

from src.tools.search import SearchProvider, search_tool
from .interface import (
    GenerateDocumentsOutput,
    GenerateSlimDocumentOutput,
    LoadConnector,
    NormalizationResult,
    PollConnector,
    SecondsSinceUnixEpoch,
    SlimConnector,
    SlimConnectorWithPermSync,
    batched,
)
from .models import Document, HierarchyNode, SlimDocument

DEFAULT_FILE_BATCH_SIZE = 32
DEFAULT_SEARCH_CONCURRENCY = 8
TEXT_FILE_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".rst",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
METADATA_KEYS = {
    "document_id",
    "doc_updated_at",
    "file_display_name",
    "link",
    "title",
}


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    return frozenset(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    )


def _expand_path(path: Path, *, glob: bool = True) -> Iterator[Path]:
    text = str(path)
    if glob and any(char in text for char in "*?[]"):
        parent = path.parent if str(path.parent) else Path(".")
        yield from parent.glob(path.name)
    elif path.is_dir():
        yield from path.rglob("*")
    else:
        yield path


class InMemoryConnector(LoadConnector):
    """Load documents from an in-memory iterable."""

    def __init__(
        self,
        documents: Iterable[Document | dict[str, Any]],
        *,
        batch_size: int = 32,
    ) -> None:
        self.documents = [self._coerce_document(document) for document in documents]
        self.batch_size = batch_size

    def load_from_state(self) -> Iterator[list[Document | HierarchyNode]]:
        yield from batched(self.documents, self.batch_size)  # type: ignore[misc]

    @staticmethod
    def _coerce_document(document: Document | dict[str, Any]) -> Document:
        if isinstance(document, Document):
            return document
        return Document(
            id=str(document["id"]),
            contents=str(document.get("contents", "")),
            title=document.get("title"),
            url=document.get("url"),
            metadata=dict(document.get("metadata") or {}),
            permissions=dict(document.get("permissions") or {}),
        )


class LocalFileConnector(LoadConnector):
    """Load local UTF-8 text files from paths, directories, or glob patterns."""

    def __init__(
        self,
        paths: Iterable[str | Path] | None = None,
        *,
        file_locations: Iterable[str | Path] | None = None,
        file_names: Iterable[str] | None = None,
        root: str | Path | None = None,
        glob: bool = True,
        encoding: str = "utf-8",
        metadata: dict[str, Any] | None = None,
        metadata_path: str | Path | None = None,
        allowed_extensions: Iterable[str] | None = None,
        batch_size: int = DEFAULT_FILE_BATCH_SIZE,
    ) -> None:
        raw_paths = paths if paths is not None else file_locations
        self.paths = [Path(path) for path in raw_paths or []]
        self.file_names = list(file_names or [])
        self.root = Path(root).expanduser().resolve() if root else None
        self.glob = glob
        self.encoding = encoding
        self.metadata = metadata or {}
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.allowed_extensions = _normalize_extensions(
            allowed_extensions or TEXT_FILE_EXTENSIONS
        )
        self.batch_size = batch_size
        self.pdf_pass: str | None = None

    def validate_connector_settings(self) -> None:
        if not self.paths:
            raise ValueError("LocalFileConnector requires at least one path.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.metadata_path and not self.metadata_path.exists():
            raise ValueError(f"metadata_path does not exist: {self.metadata_path}")

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self.pdf_pass = credentials.get("pdf_password")
        return None

    def load_from_state(self) -> Iterator[list[Document | HierarchyNode]]:
        self.validate_connector_settings()
        metadata_by_name = self._load_metadata_by_name()
        yield from batched(
            self._iter_documents(metadata_by_name),
            self.batch_size,
        )  # type: ignore[misc]

    @classmethod
    def normalize_url(cls, url: str) -> NormalizationResult:
        if not url.startswith("file://"):
            return NormalizationResult(normalized_url=None, use_default=True)
        return NormalizationResult(normalized_url=str(Path(url[7:]).resolve()))

    def _iter_documents(
        self,
        metadata_by_name: dict[str, Any],
        *,
        mtime_start: float | None = None,
        mtime_end: float | None = None,
    ) -> Iterator[Document]:
        for index, path in enumerate(self._iter_files()):
            if path.suffix.lower() not in self.allowed_extensions:
                continue
            if mtime_start is not None or mtime_end is not None:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime_start is not None and mtime < mtime_start:
                    continue
                if mtime_end is not None and mtime > mtime_end:
                    continue
            text = path.read_text(encoding=self.encoding)
            if not text.strip():
                continue
            resolved = path.resolve()
            fallback_name = (
                self.file_names[index] if index < len(self.file_names) else path.name
            )
            file_metadata = self._metadata_for_file(
                display_name=fallback_name,
                path=path,
                metadata_by_name=metadata_by_name,
            )
            processed = self._process_metadata(
                metadata=file_metadata,
                fallback_name=fallback_name,
                fallback_id=str(resolved),
                fallback_url=resolved.as_uri(),
            )
            yield Document(
                id=processed["document_id"],
                title=processed["title"],
                url=processed["link"],
                contents=text,
                metadata={
                    "source": "local_file",
                    "path": str(resolved),
                    "suffix": path.suffix,
                    "display_name": processed["file_display_name"],
                    "doc_updated_at": processed["doc_updated_at"],
                    **processed["custom_metadata"],
                },
            )

    def _iter_files(self) -> Iterator[Path]:
        seen: set[Path] = set()
        for raw_path in self.paths:
            base = (
                (self.root / raw_path)
                if self.root and not raw_path.is_absolute()
                else raw_path
            )
            for candidate in _expand_path(base, glob=self.glob):
                resolved = candidate.expanduser().resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                seen.add(resolved)
                yield resolved

    def _load_metadata_by_name(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        if self.metadata_path:
            raw = self.metadata_path.read_text(encoding=self.encoding)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                loaded.update(
                    {
                        str(item["filename"]): item
                        for item in parsed
                        if isinstance(item, dict) and "filename" in item
                    }
                )
            elif isinstance(parsed, dict):
                loaded.update(parsed)
            else:
                raise ValueError("metadata_path must contain a JSON object or list.")
        loaded.update(self.metadata)
        return loaded

    def _metadata_for_file(
        self,
        *,
        display_name: str,
        path: Path,
        metadata_by_name: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = (
            display_name,
            path.name,
            str(path),
            str(path.resolve()),
        )
        for candidate in candidates:
            value = metadata_by_name.get(candidate)
            if isinstance(value, dict):
                return dict(value)
        return {}

    def _process_metadata(
        self,
        *,
        metadata: dict[str, Any],
        fallback_name: str,
        fallback_id: str,
        fallback_url: str,
    ) -> dict[str, Any]:
        file_display_name = str(metadata.get("file_display_name") or fallback_name)
        title = str(metadata.get("title") or file_display_name)
        link = str(metadata.get("link") or fallback_url)
        updated = metadata.get("doc_updated_at")
        if isinstance(updated, datetime):
            doc_updated_at = updated.astimezone(timezone.utc).isoformat()
        elif updated:
            doc_updated_at = str(updated)
        else:
            doc_updated_at = datetime.now(timezone.utc).isoformat()

        custom_metadata = {
            key: value for key, value in metadata.items() if key not in METADATA_KEYS
        }

        return {
            "document_id": str(metadata.get("document_id") or fallback_id),
            "doc_updated_at": doc_updated_at,
            "file_display_name": file_display_name,
            "link": link,
            "title": title,
            "custom_metadata": custom_metadata,
        }


class SearchConnector(LoadConnector):
    """Load search results as documents using the existing search providers."""

    def __init__(
        self,
        queries: Iterable[str],
        *,
        provider: SearchProvider = "retrieval",
        search_url: str = "http://localhost:8000/retrieve",
        page_size: int = 5,
        batch_size: int = 32,
        max_concurrency: int = DEFAULT_SEARCH_CONCURRENCY,
    ) -> None:
        self.queries = list(queries)
        self.provider = provider
        self.search_url = search_url
        self.page_size = page_size
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    def validate_connector_settings(self) -> None:
        if not self.queries:
            raise ValueError("SearchConnector requires at least one query.")
        if self.page_size <= 0:
            raise ValueError("page_size must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero.")

    def load_from_state(self) -> Iterator[list[Document | HierarchyNode]]:
        self.validate_connector_settings()
        yield from batched(self._iter_documents_sync(), self.batch_size)  # type: ignore[misc]

    async def aload_from_state(self) -> list[Document]:
        """Async variant for callers already running inside an event loop."""

        self.validate_connector_settings()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _search_query(query: str) -> list[Document]:
            async with semaphore:
                pages = await search_tool(
                    query,
                    provider=self.provider,
                    search_url=self.search_url,
                    page_size=self.page_size,
                )
            documents: list[Document] = []
            for index, page in enumerate(pages, 1):
                if page.error:
                    continue
                doc_id = page.url or f"search:{query}:{index}"
                documents.append(
                    Document(
                        id=doc_id,
                        title=page.title,
                        url=page.url,
                        contents=page.summary,
                        metadata={"source": self.provider, "query": query},
                    )
                )
            return documents

        documents: list[Document] = []
        for query_docs in await asyncio.gather(
            *(_search_query(query) for query in self.queries)
        ):
            documents.extend(query_docs)
        return documents

    def _iter_documents_sync(self) -> Iterator[Document]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            yield from asyncio.run(self.aload_from_state())
            return
        raise RuntimeError(
            "SearchConnector.load_from_state() cannot run inside an active event "
            "loop; use await SearchConnector.aload_from_state() instead."
        )


class LocalFilePollConnector(LocalFileConnector, PollConnector):
    """Incremental sync variant of LocalFileConnector.

    poll_source yields only files whose on-disk mtime falls within [start, end].
    All other constructor options and metadata handling are inherited unchanged.
    """

    def poll_source(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
    ) -> GenerateDocumentsOutput:
        self.validate_connector_settings()
        metadata_by_name = self._load_metadata_by_name()
        yield from batched(
            self._iter_documents(metadata_by_name, mtime_start=start, mtime_end=end),
            self.batch_size,
        )  # type: ignore[misc]


class LocalFileSlimConnector(SlimConnector):
    """Enumerate local file IDs without reading content — for pruning stale documents.

    Yields SlimDocument with path and mtime metadata; no file content is loaded.
    Supports optional time-window filtering via the start/end parameters of
    retrieve_all_slim_docs.
    """

    def __init__(
        self,
        paths: Iterable[str | Path] | None = None,
        *,
        allowed_extensions: Iterable[str] | None = None,
        batch_size: int = DEFAULT_FILE_BATCH_SIZE,
    ) -> None:
        self.paths = [Path(path) for path in paths or []]
        self.allowed_extensions = _normalize_extensions(
            allowed_extensions or TEXT_FILE_EXTENSIONS
        )
        self.batch_size = batch_size

    def validate_connector_settings(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

    def retrieve_all_slim_docs(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> GenerateSlimDocumentOutput:
        self.validate_connector_settings()
        yield from batched(
            self._iter_slim_docs(start=start, end=end),
            self.batch_size,
        )  # type: ignore[misc]

    # Subclasses override this to supply permissions when _include_permissions=True.
    def _permissions_for_path(self, path: Path, stat: os.stat_result) -> dict[str, Any]:
        del path, stat
        return {}

    def _iter_slim_docs(
        self,
        *,
        start: float | None,
        end: float | None,
        _include_permissions: bool = False,
    ) -> Iterator[SlimDocument]:
        seen: set[Path] = set()
        for raw_path in self.paths:
            for candidate in _expand_path(raw_path):
                resolved = candidate.expanduser().resolve()
                if resolved in seen:
                    continue
                try:
                    st = resolved.stat()
                except OSError:
                    continue
                if not S_ISREG(st.st_mode):
                    continue
                if resolved.suffix.lower() not in self.allowed_extensions:
                    continue
                if start is not None and st.st_mtime < start:
                    continue
                if end is not None and st.st_mtime > end:
                    continue
                seen.add(resolved)
                yield SlimDocument(
                    id=str(resolved),
                    metadata={
                        "path": str(resolved),
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    },
                    permissions=(
                        self._permissions_for_path(resolved, st)
                        if _include_permissions
                        else {}
                    ),
                )


class LocalFileSlimConnectorWithPermSync(
    LocalFileSlimConnector, SlimConnectorWithPermSync
):
    """LocalFileSlimConnector that also syncs OS-level file permissions.

    Adds readable/writable/executable flags and the octal mode string to each
    SlimDocument's permissions field.  Used by permission-aware pruning pipelines.
    """

    def retrieve_all_slim_docs_perm_sync(
        self,
        start: SecondsSinceUnixEpoch | None = None,
        end: SecondsSinceUnixEpoch | None = None,
    ) -> GenerateSlimDocumentOutput:
        self.validate_connector_settings()
        yield from batched(
            self._iter_slim_docs(start=start, end=end, _include_permissions=True),
            self.batch_size,
        )  # type: ignore[misc]

    def _permissions_for_path(self, path: Path, stat: os.stat_result) -> dict[str, Any]:
        return {
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
            "executable": os.access(path, os.X_OK),
            "mode": oct(stat.st_mode & 0o777),
        }
