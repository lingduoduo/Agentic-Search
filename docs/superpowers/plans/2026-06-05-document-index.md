# Document Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `src/backend/document_index/` sample code (borrowed from the Onyx repo) from `onyx.*`/`shared_configs.*` imports to this repo's own types, create missing types and utilities, fix all import paths, and make every file importable and unit-testable.

**Architecture:** All `from onyx.*` imports are replaced with local equivalents. New types (`InferenceChunk`, `IndexFilters`, `QueryType`, `Embedding`, etc.) are added to `src/retrieval/models.py`. A new `src/backend/configs/constants.py` holds string/bool constants. A new `src/backend/document_index/utils.py` provides utility stubs. The factory is re-implemented without a DB session, using env-var flags. Backend implementations (OpenSearch, Vespa) get their import paths fixed but their business logic is left intact.

**Tech Stack:** Python 3.11+, Pydantic v2, dataclasses, stdlib only for new files.

---

## File Map

| Action | File |
|---|---|
| **Modify** | `src/retrieval/models.py` — add `QueryType`, `Embedding`, `InferenceChunk`, `InferenceChunkUncleaned`, `IndexFilters`, `MultipassConfig`, `ExternalAccess`, `DocAwareChunk`; update `IndexChunk` with enrichment fields; update `DocMetadataAwareIndexChunk` |
| **Create** | `src/backend/configs/constants.py` — `PUBLIC_DOC_PAT`, `RETURN_SEPARATOR`, `INDEX_SEPARATOR`, `SOURCE_TYPE`, Redis lock/KV constants |
| **Modify** | `src/backend/configs/app_configs.py` — add `VectorDbSettings` (Vespa/OpenSearch env vars) and wire into `AppSettings` |
| **Create** | `src/backend/document_index/utils.py` — `setup_logger`, `batch_generator`, `remove_invalid_unicode_chars`, `convert_metadata_list_of_strings_to_dict`, `get_experts_stores_representations`, `split_relationship_id` |
| **Modify** | `src/backend/document_index/vespa_constants.py` — replace `from onyx.*` with local equivalents |
| **Modify** | `src/backend/document_index/disabled.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/interfaces_new.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/document_metadata.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/document_index_utils.py` — replace `from onyx.*`; replace DB-session functions with config-based stubs; fix `chunk.source_document.id` → `chunk.embedded_chunk.chunk.document_id` |
| **Modify** | `src/backend/document_index/chunk_content_enrichment.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/factory.py` — replace DB session with env-var flags |
| **Modify** | `src/backend/document_index/opensearch/opensearch_document_index.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/opensearch/schema.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/opensearch/search.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/opensearch/client.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/vespa_document_index.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/chunk_retrieval.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/deletion.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/indexing_utils.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/kg_interactions.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/shared_utils/utils.py` — replace `from onyx.*` |
| **Modify** | `src/backend/document_index/vespa/shared_utils/vespa_request_builders.py` — replace `from onyx.*` |
| **Create** | `tests/unit/document_index/test_types.py` |
| **Create** | `tests/unit/document_index/test_disabled.py` |
| **Create** | `tests/unit/document_index/test_factory.py` |
| **Create** | `tests/unit/document_index/test_document_index_utils.py` |
| **Create** | `tests/unit/document_index/test_chunk_content_enrichment.py` |
| **Create** | `tests/unit/document_index/test_imports.py` — verify every file imports without error |

---

## Task 1: Add missing types to `src/retrieval/models.py`

**Files:**
- Modify: `src/retrieval/models.py`
- Test: `tests/unit/document_index/test_types.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/document_index/__init__.py` (empty) and `tests/unit/document_index/test_types.py`:

```python
import pytest
from datetime import datetime

from src.retrieval.models import (
    QueryType,
    Embedding,
    InferenceChunk,
    InferenceChunkUncleaned,
    IndexFilters,
    MultipassConfig,
    ExternalAccess,
)


def test_query_type_values():
    assert QueryType.SEMANTIC == "semantic"
    assert QueryType.KEYWORD == "keyword"
    assert QueryType.HYBRID == "hybrid"


def test_embedding_is_list_of_float():
    emb: Embedding = [0.1, 0.2, 0.3]
    assert isinstance(emb, list)


def test_inference_chunk_defaults():
    chunk = InferenceChunk(document_id="doc1", chunk_ind=0)
    assert chunk.blurb == ""
    assert chunk.content == ""
    assert chunk.score is None
    assert chunk.match_highlights == []
    assert chunk.document_sets == set()


def test_inference_chunk_full():
    chunk = InferenceChunk(
        document_id="doc1",
        chunk_ind=2,
        blurb="blurb text",
        content="full content",
        score=0.95,
        source_type="web",
        match_highlights=["highlight"],
    )
    assert chunk.content == "full content"
    assert chunk.score == 0.95


def test_inference_chunk_uncleaned_to_inference_chunk():
    raw = InferenceChunkUncleaned(
        document_id="doc1",
        chunk_ind=0,
        content="Title\n\nReal content",
        title="Title",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    cleaned = raw.to_inference_chunk()
    assert isinstance(cleaned, InferenceChunk)
    assert cleaned.document_id == "doc1"


def test_index_filters_defaults():
    f = IndexFilters()
    assert f.access_control_list is None
    assert f.document_set is None
    assert f.source_type is None
    assert f.time_cutoff is None


def test_index_filters_with_values():
    f = IndexFilters(
        access_control_list=["user1", "PUBLIC"],
        document_set=["set1"],
        source_type=["web"],
        time_cutoff=datetime(2024, 1, 1),
    )
    assert "PUBLIC" in f.access_control_list
    assert f.document_set == ["set1"]


def test_multipass_config_defaults():
    cfg = MultipassConfig()
    assert cfg.multipass_indexing is False
    assert cfg.enable_large_chunks is False


def test_external_access_defaults():
    ea = ExternalAccess()
    assert ea.external_user_emails == []
    assert ea.external_user_group_ids == []
    assert ea.is_public is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/document_index/test_types.py -v
```

Expected: FAIL with `ImportError` (types not defined yet)

- [ ] **Step 3: Add the new types to `src/retrieval/models.py`**

At the top of `src/retrieval/models.py`, add `from datetime import datetime` if not present.

After `class EmbeddingPrecision`, add:

```python
class QueryType(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# Type alias matching shared_configs.model_server_models.Embedding
Embedding = list[float]
```

After `class DocumentAccess`, add:

```python
@dataclass
class ExternalAccess:
    """External (third-party) access control metadata for a document."""

    external_user_emails: list[str] = field(default_factory=list)
    external_user_group_ids: list[str] = field(default_factory=list)
    is_public: bool = False
```

After `class DocMetadataAwareIndexChunk`, add:

```python
@dataclass(frozen=True)
class MultipassConfig:
    """Configuration for multipass indexing (large + mini chunks)."""

    multipass_indexing: bool = False
    enable_large_chunks: bool = False
```

After `MultipassConfig`, add the Pydantic `InferenceChunk` classes. Since the file already conditionally imports pydantic, add inside the `try` block (after `SearchDocsResponse`) or after it unconditionally with a guard. Add at the **end** of the file, after the existing Pydantic block:

```python
try:
    from pydantic import BaseModel as _PydanticBase2
    from pydantic import Field as _Field2
    from datetime import datetime as _dt

    class InferenceChunk(_PydanticBase2):
        """A retrieved chunk returned from a DocumentIndex retrieval method."""

        document_id: str
        chunk_ind: int
        blurb: str = ""
        content: str = ""
        source_links: dict[int, str] | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        score: float | None = None
        metadata: dict[str, Any] = _Field2(default_factory=dict)
        match_highlights: list[str] = _Field2(default_factory=list)
        document_sets: set[str] = _Field2(default_factory=set)
        access_control_list: list[str] | None = None
        title: str | None = None
        source_type: str = ""
        large_chunk_id: int | None = None
        large_chunk_reference_ids: list[int] | None = None

    class InferenceChunkUncleaned(_PydanticBase2):
        """Mutable InferenceChunk used during content cleanup pipeline."""

        document_id: str
        chunk_ind: int
        content: str = ""
        title: str | None = None
        blurb: str = ""
        source_links: dict[int, str] | None = None
        section_continuation: bool = False
        semantic_identifier: str = ""
        boost: int = 0
        hidden: bool = False
        score: float | None = None
        metadata: dict[str, Any] = _Field2(default_factory=dict)
        match_highlights: list[str] = _Field2(default_factory=list)
        document_sets: set[str] = _Field2(default_factory=set)
        access_control_list: list[str] | None = None
        source_type: str = ""
        metadata_suffix: str = ""
        doc_summary: str = ""
        chunk_context: str = ""

        def to_inference_chunk(self) -> "InferenceChunk":
            return InferenceChunk(
                document_id=self.document_id,
                chunk_ind=self.chunk_ind,
                blurb=self.blurb,
                content=self.content,
                source_links=self.source_links,
                section_continuation=self.section_continuation,
                semantic_identifier=self.semantic_identifier,
                boost=self.boost,
                hidden=self.hidden,
                score=self.score,
                metadata=self.metadata,
                match_highlights=self.match_highlights,
                document_sets=self.document_sets,
                access_control_list=self.access_control_list,
                source_type=self.source_type,
            )

    class IndexFilters(_PydanticBase2):
        """Filters passed to DocumentIndex retrieval methods."""

        model_config = {"frozen": True}

        access_control_list: list[str] | None = None
        document_set: list[str] | None = None
        source_type: list[str] | None = None
        tags: dict[str, list[str]] | None = None
        time_cutoff: _dt | None = None
        is_public: bool | None = None
        tenant_id: str | None = None

except ImportError:  # pydantic not available

    @dataclass  # type: ignore[no-redef]
    class InferenceChunk:  # type: ignore[no-redef]
        document_id: str
        chunk_ind: int
        blurb: str = ""
        content: str = ""
        score: float | None = None
        metadata: dict = field(default_factory=dict)
        match_highlights: list = field(default_factory=list)
        document_sets: set = field(default_factory=set)
        access_control_list: list | None = None
        title: str | None = None
        source_type: str = ""

    @dataclass  # type: ignore[no-redef]
    class InferenceChunkUncleaned:  # type: ignore[no-redef]
        document_id: str
        chunk_ind: int
        content: str = ""
        title: str | None = None
        metadata_suffix: str = ""
        doc_summary: str = ""
        chunk_context: str = ""

        def to_inference_chunk(self) -> "InferenceChunk":
            return InferenceChunk(
                document_id=self.document_id,
                chunk_ind=self.chunk_ind,
                content=self.content,
            )

    @dataclass  # type: ignore[no-redef]
    class IndexFilters:  # type: ignore[no-redef]
        access_control_list: list | None = None
        document_set: list | None = None
        source_type: list | None = None
        time_cutoff: Any = None
        is_public: bool | None = None
        tenant_id: str | None = None
```

Also update `IndexChunk` to add enrichment fields (add after `large_chunk_id`):

```python
    title_prefix: str = ""
    doc_summary: str = ""
    chunk_context: str = ""
```

And update `DocMetadataAwareIndexChunk` to expose enrichment fields as properties (add after `ancestor_hierarchy_node_ids`):

```python
    @property
    def title_prefix(self) -> str:
        return self.embedded_chunk.chunk.title_prefix

    @property
    def doc_summary(self) -> str:
        return self.embedded_chunk.chunk.doc_summary

    @property
    def chunk_context(self) -> str:
        return self.embedded_chunk.chunk.chunk_context

    @property
    def content(self) -> str:
        return self.embedded_chunk.chunk.text

    @property
    def metadata_suffix_keyword(self) -> str:
        return self.embedded_chunk.chunk.metadata_suffix_keyword

    @property
    def metadata_suffix_semantic(self) -> str:
        return self.embedded_chunk.chunk.metadata_suffix_semantic

    @property
    def chunk_id(self) -> int:
        return self.embedded_chunk.chunk.chunk_id

    @property
    def large_chunk_id(self) -> int | None:
        return self.embedded_chunk.chunk.large_chunk_id
```

Add `DocAwareChunk` as an alias at the bottom of the file:

```python
# Alias used by chunk_content_enrichment.py
DocAwareChunk = DocMetadataAwareIndexChunk
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/document_index/test_types.py -v
```

Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/models.py tests/unit/document_index/
git commit -m "feat(models): add InferenceChunk, IndexFilters, QueryType, Embedding, MultipassConfig, ExternalAccess for document index"
```

---

## Task 2: Add config constants and vector-DB env vars

**Files:**
- Create: `src/backend/configs/constants.py`
- Modify: `src/backend/configs/app_configs.py`
- Test: `tests/unit/document_index/test_types.py` (extend)

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/document_index/test_types.py`:

```python
def test_constants_importable():
    from src.backend.configs.constants import (
        PUBLIC_DOC_PAT,
        RETURN_SEPARATOR,
        INDEX_SEPARATOR,
        SOURCE_TYPE,
    )
    assert PUBLIC_DOC_PAT == "PUBLIC"
    assert isinstance(RETURN_SEPARATOR, str)
    assert isinstance(INDEX_SEPARATOR, str)
    assert SOURCE_TYPE == "source_type"


def test_vector_db_settings_defaults():
    from src.backend.configs.app_configs import VectorDbSettings
    s = VectorDbSettings()
    assert s.disable_vector_db is False
    assert s.disable_vespa is False
    assert s.enable_opensearch is False
    assert "localhost" in s.vespa_host
    assert "localhost" in s.opensearch_host


def test_vector_db_settings_from_env():
    from src.backend.configs.app_configs import VectorDbSettings, get_env_bool, get_env_str
    # Just verify the class can be constructed with all fields
    s = VectorDbSettings(disable_vector_db=True)
    assert s.disable_vector_db is True
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_types.py::test_constants_importable tests/unit/document_index/test_types.py::test_vector_db_settings_defaults -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/backend/configs/constants.py`**

```python
"""String constants shared by the document index and retrieval layers.

These replace direct imports from onyx.configs.constants.
"""

# Represents a document accessible to all users (no access restriction).
PUBLIC_DOC_PAT = "PUBLIC"

# Separator used when joining multi-part content (title + body, etc.)
RETURN_SEPARATOR = "\n\n"

# Separator used when building index names
INDEX_SEPARATOR = "__"

# Field name for source type in Vespa YQL and OpenSearch schema
SOURCE_TYPE = "source_type"

# Key for the reindex flag in the key-value store
KV_REINDEX_KEY = "kv_reindex_key"

# Redis lock name for Vespa reindex coordination
VESPA_REINDEX_REDIS_LOCK = "vespa_reindex_lock"

# Blurb size used for title prefix matching during content cleanup
BLURB_SIZE = 250

# OpenSearch migration constants
OPENSEARCH_MIGRATION_ENABLED_KEY = "opensearch_migration_enabled"
OPENSEARCH_RETRIEVAL_ENABLED_KEY = "opensearch_retrieval_enabled"
```

- [ ] **Step 4: Add `VectorDbSettings` to `src/backend/configs/app_configs.py`**

Add after `class LLMSettings`:

```python
@dataclass(frozen=True)
class VectorDbSettings:
    """Settings for Vespa and OpenSearch vector database backends."""

    disable_vector_db: bool = False
    disable_vespa: bool = False
    enable_opensearch: bool = False
    multi_tenant: bool = False

    # Vespa connection
    vespa_host: str = "localhost"
    vespa_port: int = 8080
    vespa_tenant_port: int = 19071
    vespa_cloud_url: str | None = None
    vespa_timeout: str = "10s"
    vespa_language_override: str | None = None
    vespa_searcher_threads: int = 8

    # OpenSearch connection
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str | None = None
    opensearch_password: str | None = None

    # Indexing behaviour
    max_chunks_per_doc_batch: int = 512
    enable_multipass_indexing: bool = False
    verify_create_opensearch_index_on_init: bool = False
```

Add `vector_db: VectorDbSettings = field(default_factory=VectorDbSettings)` to `AppSettings`.

Add to `load_app_settings`:

```python
        vector_db=VectorDbSettings(
            disable_vector_db=get_env_bool(source, "DISABLE_VECTOR_DB", False),
            disable_vespa=get_env_bool(source, "ONYX_DISABLE_VESPA", False),
            enable_opensearch=get_env_bool(source, "ENABLE_OPENSEARCH_INDEXING_FOR_ONYX", False),
            multi_tenant=get_env_bool(source, "MULTI_TENANT", False),
            vespa_host=get_env_str(source, "VESPA_HOST", "localhost"),
            vespa_port=get_env_int(source, "VESPA_PORT", 8080),
            vespa_tenant_port=get_env_int(source, "VESPA_TENANT_PORT", 19071),
            vespa_cloud_url=get_env_str(source, "VESPA_CLOUD_URL", None),
            vespa_timeout=get_env_str(source, "VESPA_TIMEOUT", "10s"),
            vespa_language_override=get_env_str(source, "VESPA_LANGUAGE_OVERRIDE", None),
            vespa_searcher_threads=get_env_int(source, "VESPA_SEARCHER_THREADS", 8),
            opensearch_host=get_env_str(source, "OPENSEARCH_HOST", "localhost"),
            opensearch_port=get_env_int(source, "OPENSEARCH_PORT", 9200),
            opensearch_user=get_env_str(source, "OPENSEARCH_USER", None),
            opensearch_password=get_env_str(source, "OPENSEARCH_PASSWORD", None),
            max_chunks_per_doc_batch=get_env_int(source, "MAX_CHUNKS_PER_DOC_BATCH", 512),
            enable_multipass_indexing=get_env_bool(source, "ENABLE_MULTIPASS_INDEXING", False),
            verify_create_opensearch_index_on_init=get_env_bool(
                source, "VERIFY_CREATE_OPENSEARCH_INDEX_ON_INIT_MT", False
            ),
        ),
```

Add `VectorDbSettings` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/document_index/test_types.py -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/backend/configs/constants.py src/backend/configs/app_configs.py tests/unit/document_index/test_types.py
git commit -m "feat(config): add constants.py and VectorDbSettings for document index backends"
```

---

## Task 3: Create utility helpers module

**Files:**
- Create: `src/backend/document_index/utils.py`
- Test: `tests/unit/document_index/test_types.py` (extend)

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/document_index/test_types.py`:

```python
def test_utils_importable():
    from src.backend.document_index.utils import (
        setup_logger,
        batch_generator,
        remove_invalid_unicode_chars,
        convert_metadata_list_of_strings_to_dict,
        get_experts_stores_representations,
        split_relationship_id,
    )


def test_batch_generator():
    from src.backend.document_index.utils import batch_generator
    items = list(range(10))
    batches = list(batch_generator(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batch_generator_exact():
    from src.backend.document_index.utils import batch_generator
    batches = list(batch_generator([1, 2, 3], 3))
    assert batches == [[1, 2, 3]]


def test_remove_invalid_unicode_chars():
    from src.backend.document_index.utils import remove_invalid_unicode_chars
    assert remove_invalid_unicode_chars("hello\x00world") == "helloworld"
    assert remove_invalid_unicode_chars("normal text") == "normal text"


def test_convert_metadata_list_of_strings_to_dict():
    from src.backend.document_index.utils import convert_metadata_list_of_strings_to_dict
    result = convert_metadata_list_of_strings_to_dict(["key1:val1", "key2:val2"])
    assert result == {"key1": "val1", "key2": "val2"}


def test_convert_metadata_dict_passthrough():
    from src.backend.document_index.utils import convert_metadata_list_of_strings_to_dict
    result = convert_metadata_list_of_strings_to_dict({"key": "val"})
    assert result == {"key": "val"}


def test_split_relationship_id():
    from src.backend.document_index.utils import split_relationship_id
    source, rel, target = split_relationship_id("doc1:RELATED:doc2")
    assert source == "doc1"
    assert rel == "RELATED"
    assert target == "doc2"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_types.py::test_utils_importable tests/unit/document_index/test_types.py::test_batch_generator -v
```

Expected: FAIL

- [ ] **Step 3: Create `src/backend/document_index/utils.py`**

```python
"""Utility helpers for the document index layer.

Replaces imports previously satisfied by onyx.utils.* and onyx.connectors.*.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator, Iterable
from typing import Any, TypeVar

_T = TypeVar("_T")


def setup_logger(name: str) -> logging.Logger:
    """Drop-in replacement for onyx.utils.logger.setup_logger."""
    return logging.getLogger(name)


def batch_generator(
    items: Iterable[_T], batch_size: int
) -> Generator[list[_T], None, None]:
    """Yield successive non-overlapping batches from items."""
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


_INVALID_UNICODE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def remove_invalid_unicode_chars(text: str) -> str:
    """Remove control characters that are invalid in most storage backends."""
    return _INVALID_UNICODE_RE.sub("", text)


def convert_metadata_list_of_strings_to_dict(
    metadata: list[str] | dict[str, Any],
) -> dict[str, Any]:
    """Convert a list of 'key:value' strings to a dict, or pass through a dict."""
    if isinstance(metadata, dict):
        return metadata
    result: dict[str, Any] = {}
    for item in metadata:
        if ":" in item:
            key, _, value = item.partition(":")
            result[key] = value
    return result


def get_experts_stores_representations(
    primary_owners: list[str] | None,
    secondary_owners: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return expert email lists. Stub — returns inputs unchanged."""
    return primary_owners or [], secondary_owners or []


def split_relationship_id(relationship_id: str) -> tuple[str, str, str]:
    """Split a 'source:RELATION:target' relationship ID string.

    Returns (source, relation, target). Raises ValueError if format is wrong.
    """
    parts = relationship_id.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"relationship_id must be 'source:RELATION:target', got: {relationship_id!r}"
        )
    return parts[0], parts[1], parts[2]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/document_index/test_types.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backend/document_index/utils.py tests/unit/document_index/test_types.py
git commit -m "feat(document_index): add utils.py with logger, batch, unicode, metadata helpers"
```

---

## Task 4: Fix simple support files

**Files:**
- Modify: `src/backend/document_index/vespa_constants.py`
- Modify: `src/backend/document_index/disabled.py`
- Modify: `src/backend/document_index/interfaces_new.py`
- Modify: `src/backend/document_index/document_metadata.py`
- Test: `tests/unit/document_index/test_imports.py`

- [ ] **Step 1: Write failing import tests**

Create `tests/unit/document_index/test_imports.py`:

```python
"""Verify every document_index file can be imported without error."""


def test_vespa_constants_importable():
    import src.backend.document_index.vespa_constants  # noqa: F401


def test_disabled_importable():
    import src.backend.document_index.disabled  # noqa: F401


def test_interfaces_new_importable():
    import src.backend.document_index.interfaces_new  # noqa: F401


def test_document_metadata_importable():
    import src.backend.document_index.document_metadata  # noqa: F401
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_imports.py -v
```

Expected: 4 FAIL with `ModuleNotFoundError: No module named 'onyx'`

- [ ] **Step 3: Fix `src/backend/document_index/vespa_constants.py`**

Replace the first 6 lines (the onyx imports) with:

```python
import os

from src.backend.configs.constants import SOURCE_TYPE  # noqa: F401 — re-exported

_vespa_cloud_url = os.environ.get("VESPA_CLOUD_URL") or None
_vespa_config_server_host = os.environ.get("VESPA_CONFIG_SERVER_HOST", "localhost")
_vespa_tenant_port = int(os.environ.get("VESPA_TENANT_PORT", "19071"))
_vespa_host = os.environ.get("VESPA_HOST", "localhost")
_vespa_port = int(os.environ.get("VESPA_PORT", "8080"))

VESPA_CLOUD_URL = _vespa_cloud_url
VESPA_CONFIG_SERVER_HOST = _vespa_config_server_host
VESPA_HOST = _vespa_host
VESPA_PORT = _vespa_port
VESPA_TENANT_PORT = _vespa_tenant_port
```

- [ ] **Step 4: Fix `src/backend/document_index/disabled.py`**

Replace all `from onyx.*` imports with:

```python
from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import EmbeddingPrecision
from src.retrieval.models import Embedding
from src.retrieval.models import IndexFilters
from src.retrieval.models import InferenceChunk
from src.retrieval.models import QueryType
from src.backend.document_index.interfaces_new import DocumentIndex
from src.backend.document_index.interfaces_new import DocumentInsertionRecord
from src.backend.document_index.interfaces_new import DocumentSectionRequest
from src.backend.document_index.interfaces_new import IndexingMetadata
from src.backend.document_index.interfaces_new import MetadataUpdateRequest
```

- [ ] **Step 5: Fix `src/backend/document_index/interfaces_new.py`**

Replace all `from onyx.*` and `from shared_configs.*` imports with:

```python
from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import DocumentAccess
from src.retrieval.models import EmbeddingPrecision
from src.retrieval.models import Embedding
from src.retrieval.models import IndexFilters
from src.retrieval.models import InferenceChunk
from src.retrieval.models import QueryType
from src.backend.configs.constants import PUBLIC_DOC_PAT
from src.backend.document_index.opensearch.constants import DEFAULT_MAX_CHUNK_SIZE
```

- [ ] **Step 6: Fix `src/backend/document_index/document_metadata.py`**

Replace `from onyx.access.models import ExternalAccess` with:

```python
from src.retrieval.models import ExternalAccess
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/unit/document_index/test_imports.py -v
```

Expected: All 4 PASS

- [ ] **Step 8: Commit**

```bash
git add src/backend/document_index/vespa_constants.py src/backend/document_index/disabled.py \
    src/backend/document_index/interfaces_new.py src/backend/document_index/document_metadata.py \
    tests/unit/document_index/test_imports.py
git commit -m "fix(document_index): replace onyx imports in vespa_constants, disabled, interfaces_new, document_metadata"
```

---

## Task 5: Fix `document_index_utils.py` and `chunk_content_enrichment.py`

**Files:**
- Modify: `src/backend/document_index/document_index_utils.py`
- Modify: `src/backend/document_index/chunk_content_enrichment.py`
- Test: `tests/unit/document_index/test_document_index_utils.py`
- Test: `tests/unit/document_index/test_chunk_content_enrichment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/document_index/test_document_index_utils.py`:

```python
import math
import pytest
from uuid import UUID

from src.backend.document_index.document_index_utils import (
    translate_boost_count_to_multiplier,
    get_uuid_from_chunk_info,
    get_uuid_from_chunk_info_old,
    should_use_multipass,
    DEFAULT_BATCH_SIZE,
)


def test_boost_zero_is_one():
    assert translate_boost_count_to_multiplier(0) == pytest.approx(1.0, abs=0.01)


def test_boost_negative_below_one():
    result = translate_boost_count_to_multiplier(-10)
    assert 0.5 < result < 1.0


def test_boost_positive_above_one():
    result = translate_boost_count_to_multiplier(10)
    assert 1.0 < result <= 2.0


def test_get_uuid_from_chunk_info_returns_uuid():
    uid = get_uuid_from_chunk_info(
        document_id="doc1", chunk_id=0, tenant_id="tenant1"
    )
    assert isinstance(uid, UUID)


def test_get_uuid_from_chunk_info_deterministic():
    uid1 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    uid2 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    assert uid1 == uid2


def test_get_uuid_from_chunk_info_trailing_slash():
    uid1 = get_uuid_from_chunk_info(document_id="doc1/", chunk_id=0, tenant_id="t1")
    uid2 = get_uuid_from_chunk_info(document_id="doc1", chunk_id=0, tenant_id="t1")
    assert uid1 == uid2


def test_get_uuid_from_chunk_info_old():
    uid = get_uuid_from_chunk_info_old(document_id="doc1", chunk_id=0)
    assert isinstance(uid, UUID)


def test_should_use_multipass_none_uses_default():
    result = should_use_multipass(None)
    assert isinstance(result, bool)


def test_default_batch_size():
    assert DEFAULT_BATCH_SIZE == 30
```

Create `tests/unit/document_index/test_chunk_content_enrichment.py`:

```python
from src.retrieval.models import InferenceChunkUncleaned
from src.backend.document_index.chunk_content_enrichment import cleanup_content_for_chunks


def test_cleanup_strips_title():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="My Title\n\nActual content here.",
        title="My Title",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert len(result) == 1
    assert "My Title" not in result[0].content
    assert "Actual content here." in result[0].content


def test_cleanup_strips_metadata_suffix():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="Main content\n\ntag:value",
        metadata_suffix="tag:value",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert "tag:value" not in result[0].content


def test_cleanup_no_title_noop():
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="Just content",
        metadata_suffix="",
        doc_summary="",
        chunk_context="",
    )
    result = cleanup_content_for_chunks([chunk])
    assert result[0].content == "Just content"


def test_cleanup_returns_inference_chunk_type():
    from src.retrieval.models import InferenceChunk
    chunk = InferenceChunkUncleaned(
        document_id="d1",
        chunk_ind=0,
        content="content",
    )
    result = cleanup_content_for_chunks([chunk])
    assert isinstance(result[0], InferenceChunk)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_document_index_utils.py tests/unit/document_index/test_chunk_content_enrichment.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Fix `src/backend/document_index/document_index_utils.py`**

Replace the import block at the top (lines 1-15) with:

```python
import math
import os
import uuid
from uuid import UUID

from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import MultipassConfig
from src.backend.document_index.vespa.internal_types import EnrichedDocumentIndexingInfo

MULTI_TENANT: bool = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
ENABLE_MULTIPASS_INDEXING: bool = (
    os.environ.get("ENABLE_MULTIPASS_INDEXING", "").lower() in {"1", "true", "yes"}
)

DEFAULT_BATCH_SIZE = 30
DEFAULT_INDEX_NAME = "danswer_chunk"
```

Replace the three functions that use `SearchSettings` / DB session:

```python
def should_use_multipass(search_settings: "IndexSearchSettings | None") -> bool:
    """Determine multipass usage from settings or env default."""
    if search_settings is not None:
        return getattr(search_settings, "multipass_indexing", False)
    return ENABLE_MULTIPASS_INDEXING


def get_multipass_config(search_settings: "IndexSearchSettings") -> MultipassConfig:
    """Build a MultipassConfig from search settings."""
    multipass = should_use_multipass(search_settings)
    enable_large_chunks = getattr(search_settings, "large_chunks_enabled", False)
    return MultipassConfig(
        multipass_indexing=multipass, enable_large_chunks=enable_large_chunks
    )


def get_both_index_properties(
    primary_index_name: str,
    secondary_index_name: str | None = None,
    primary_multipass: bool = False,
    secondary_multipass: bool = False,
) -> tuple[str, str | None, bool, bool | None]:
    """Return index names and multipass flags (no DB session required)."""
    return (
        primary_index_name,
        secondary_index_name,
        primary_multipass,
        secondary_multipass if secondary_index_name else None,
    )
```

Add the `IndexSearchSettings` TypedDict after the imports to satisfy type hints:

```python
from typing import Protocol

class IndexSearchSettings(Protocol):
    multipass_indexing: bool
    large_chunks_enabled: bool
    index_name: str
```

Fix `get_uuid_from_chunk` to use `embedded_chunk.chunk.document_id`:

```python
def get_uuid_from_chunk(chunk: DocMetadataAwareIndexChunk) -> uuid.UUID:
    return get_uuid_from_chunk_info(
        document_id=chunk.embedded_chunk.chunk.document_id,
        chunk_id=chunk.embedded_chunk.chunk.chunk_id,
        tenant_id=chunk.tenant_id,
        large_chunk_id=chunk.embedded_chunk.chunk.large_chunk_id,
    )


def get_uuid_from_chunk_old(
    chunk: DocMetadataAwareIndexChunk, large_chunk_reference_ids: list[int] = []
) -> UUID:
    return get_uuid_from_chunk_info_old(
        document_id=chunk.embedded_chunk.chunk.document_id,
        chunk_id=chunk.embedded_chunk.chunk.chunk_id,
        large_chunk_reference_ids=large_chunk_reference_ids,
    )
```

Fix the `MULTI_TENANT` usage in `get_uuid_from_chunk_info`:

```python
    if MULTI_TENANT:
        unique_identifier_string += "_" + tenant_id
```

(This is already correct once the import is fixed.)

- [ ] **Step 4: Fix `src/backend/document_index/chunk_content_enrichment.py`**

Replace the entire import block with:

```python
import os
from src.backend.configs.constants import BLURB_SIZE, RETURN_SEPARATOR
from src.retrieval.models import InferenceChunk
from src.retrieval.models import InferenceChunkUncleaned
from src.retrieval.models import DocAwareChunk
from src.retrieval.models import DocMetadataAwareIndexChunk
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/document_index/test_document_index_utils.py tests/unit/document_index/test_chunk_content_enrichment.py -v
```

Expected: All PASS

- [ ] **Step 6: Update import tests**

Add to `tests/unit/document_index/test_imports.py`:

```python
def test_document_index_utils_importable():
    import src.backend.document_index.document_index_utils  # noqa: F401


def test_chunk_content_enrichment_importable():
    import src.backend.document_index.chunk_content_enrichment  # noqa: F401
```

Run: `pytest tests/unit/document_index/test_imports.py -v` — all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/backend/document_index/document_index_utils.py src/backend/document_index/chunk_content_enrichment.py \
    tests/unit/document_index/
git commit -m "fix(document_index): replace onyx imports in document_index_utils and chunk_content_enrichment"
```

---

## Task 6: Fix `factory.py`

**Files:**
- Modify: `src/backend/document_index/factory.py`
- Test: `tests/unit/document_index/test_factory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/document_index/test_factory.py`:

```python
import pytest


def test_factory_importable():
    import src.backend.document_index.factory  # noqa: F401


def test_get_default_index_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    import importlib
    import src.backend.document_index.factory as factory_mod
    importlib.reload(factory_mod)
    from src.backend.document_index.disabled import DisabledDocumentIndex

    idx = factory_mod.get_default_document_index(
        primary_index_name="test_index",
        secondary_index_name=None,
    )
    assert isinstance(idx, DisabledDocumentIndex)
    # Should be a no-op (not raise)
    idx.verify_and_create_index_if_necessary(
        embedding_dim=768, embedding_precision=None
    )


def test_get_default_index_disabled_returns_list(monkeypatch):
    monkeypatch.setenv("DISABLE_VECTOR_DB", "true")
    import importlib
    import src.backend.document_index.factory as factory_mod
    importlib.reload(factory_mod)
    from src.backend.document_index.disabled import DisabledDocumentIndex

    indices = factory_mod.get_all_document_indices(
        primary_index_name="test_index",
        secondary_index_name=None,
    )
    assert len(indices) == 1
    assert isinstance(indices[0], DisabledDocumentIndex)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_factory.py -v
```

Expected: FAIL

- [ ] **Step 3: Rewrite `src/backend/document_index/factory.py`**

Replace the entire file with:

```python
"""Factory for creating DocumentIndex instances.

Selects the appropriate backend (Vespa, OpenSearch, or Disabled) from env vars.
No DB session required — backend selection is purely config-driven.
"""

import os

from src.backend.document_index.disabled import DisabledDocumentIndex
from src.backend.document_index.interfaces_new import DocumentIndex

_DISABLE_VECTOR_DB = os.environ.get("DISABLE_VECTOR_DB", "").lower() in {
    "1", "true", "yes"
}
_ONYX_DISABLE_VESPA = os.environ.get("ONYX_DISABLE_VESPA", "").lower() in {
    "1", "true", "yes"
}
_ENABLE_OPENSEARCH = os.environ.get("ENABLE_OPENSEARCH_INDEXING_FOR_ONYX", "").lower() in {
    "1", "true", "yes"
}


def get_default_document_index(
    primary_index_name: str,
    secondary_index_name: str | None,
    large_chunks_enabled: bool = False,
    secondary_large_chunks_enabled: bool = False,
    embedding_dim: int = 768,
    secondary_embedding_dim: int | None = None,
) -> DocumentIndex:
    """Get the primary document index for retrieval.

    Returns DisabledDocumentIndex when DISABLE_VECTOR_DB=true.
    Returns OpenSearchDocumentIndex when ENABLE_OPENSEARCH_INDEXING_FOR_ONYX=true.
    Otherwise returns VespaDocumentIndex.
    """
    if _DISABLE_VECTOR_DB:
        return DisabledDocumentIndex()

    if _ENABLE_OPENSEARCH:
        from src.backend.document_index.opensearch.opensearch_document_index import (
            OpenSearchDocumentIndex,
            OpenSearchIndexPair,
        )
        from src.backend.document_index.interfaces_new import TenantState

        tenant_state = _build_tenant_state()
        primary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=primary_index_name,
            embedding_dim=embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        if secondary_index_name is None:
            return OpenSearchIndexPair(primary=primary, secondary=None)
        secondary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=secondary_index_name,
            embedding_dim=secondary_embedding_dim or embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        return OpenSearchIndexPair(
            primary=primary,
            secondary=secondary,
            secondary_embedding_dim=secondary_embedding_dim or embedding_dim,
            secondary_embedding_precision=_get_embedding_precision(),
        )

    from src.backend.document_index.vespa.vespa_document_index import (
        VespaDocumentIndex,
        VespaIndexPair,
    )

    tenant_state = _build_tenant_state()
    primary = VespaDocumentIndex(
        index_name=primary_index_name,
        tenant_state=_build_tenant_state(),
        large_chunks_enabled=large_chunks_enabled,
    )
    if secondary_index_name is None:
        return VespaIndexPair(
            primary=primary,
            secondary=None,
            secondary_index_name=None,
            secondary_embedding_dim=None,
            secondary_embedding_precision=None,
        )
    secondary = VespaDocumentIndex(
        index_name=secondary_index_name,
        tenant_state=tenant_state,
        large_chunks_enabled=secondary_large_chunks_enabled,
    )
    return VespaIndexPair(
        primary=primary,
        secondary=secondary,
        secondary_index_name=secondary_index_name,
        secondary_embedding_dim=secondary_embedding_dim,
        secondary_embedding_precision=_get_embedding_precision(),
    )


def get_all_document_indices(
    primary_index_name: str,
    secondary_index_name: str | None,
    large_chunks_enabled: bool = False,
    secondary_large_chunks_enabled: bool = False,
    embedding_dim: int = 768,
    secondary_embedding_dim: int | None = None,
) -> list[DocumentIndex]:
    """Get every document index that should be written to during indexing."""
    if _DISABLE_VECTOR_DB:
        return [DisabledDocumentIndex()]

    if _ONYX_DISABLE_VESPA and not _ENABLE_OPENSEARCH:
        raise ValueError(
            "ONYX_DISABLE_VESPA is set but ENABLE_OPENSEARCH_INDEXING_FOR_ONYX is not."
        )

    result: list[DocumentIndex] = []
    if not _ONYX_DISABLE_VESPA:
        result.append(
            get_default_document_index(
                primary_index_name=primary_index_name,
                secondary_index_name=secondary_index_name,
                large_chunks_enabled=large_chunks_enabled,
                secondary_large_chunks_enabled=secondary_large_chunks_enabled,
                embedding_dim=embedding_dim,
                secondary_embedding_dim=secondary_embedding_dim,
            )
        )
    if _ENABLE_OPENSEARCH:
        from src.backend.document_index.opensearch.opensearch_document_index import (
            OpenSearchDocumentIndex,
            OpenSearchIndexPair,
        )
        tenant_state = _build_tenant_state()
        primary = OpenSearchDocumentIndex(
            tenant_state=tenant_state,
            index_name=primary_index_name,
            embedding_dim=embedding_dim,
            embedding_precision=_get_embedding_precision(),
        )
        secondary = None
        if secondary_index_name:
            secondary = OpenSearchDocumentIndex(
                tenant_state=tenant_state,
                index_name=secondary_index_name,
                embedding_dim=secondary_embedding_dim or embedding_dim,
                embedding_precision=_get_embedding_precision(),
            )
        result.append(OpenSearchIndexPair(primary=primary, secondary=secondary))
    return result


def _build_tenant_state():
    from src.backend.document_index.interfaces_new import TenantState
    tenant_id = os.environ.get("CURRENT_TENANT_ID", "default")
    multi_tenant = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
    return TenantState(tenant_id=tenant_id, multitenant=multi_tenant)


def _get_embedding_precision():
    from src.retrieval.models import EmbeddingPrecision
    raw = os.environ.get("EMBEDDING_PRECISION", "float")
    return EmbeddingPrecision(raw) if raw in ("float", "bfloat16") else EmbeddingPrecision.FLOAT
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/document_index/test_factory.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/backend/document_index/factory.py tests/unit/document_index/test_factory.py
git commit -m "fix(document_index): rewrite factory.py — remove DB session, use env-var backend selection"
```

---

## Task 7: Fix OpenSearch implementation imports

**Files:**
- Modify: `src/backend/document_index/opensearch/opensearch_document_index.py`
- Modify: `src/backend/document_index/opensearch/schema.py`
- Modify: `src/backend/document_index/opensearch/search.py`
- Modify: `src/backend/document_index/opensearch/client.py`
- Test: `tests/unit/document_index/test_imports.py` (extend)

- [ ] **Step 1: Write failing import tests**

Add to `tests/unit/document_index/test_imports.py`:

```python
def test_opensearch_constants_importable():
    import src.backend.document_index.opensearch.constants  # noqa: F401


def test_opensearch_schema_importable():
    import src.backend.document_index.opensearch.schema  # noqa: F401


def test_opensearch_search_importable():
    import src.backend.document_index.opensearch.search  # noqa: F401


def test_opensearch_client_importable():
    import src.backend.document_index.opensearch.client  # noqa: F401


def test_opensearch_document_index_importable():
    import src.backend.document_index.opensearch.opensearch_document_index  # noqa: F401
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_imports.py::test_opensearch_schema_importable -v
```

Expected: FAIL

- [ ] **Step 3: Fix imports in `opensearch/schema.py`**

Find all `from onyx.*` lines and apply this substitution table:

| Old import | New import |
|---|---|
| `from onyx.access.models import DocumentAccess` | `from src.retrieval.models import DocumentAccess` |
| `from onyx.context.search.models import InferenceChunk` | `from src.retrieval.models import InferenceChunk` |
| `from onyx.context.search.models import InferenceChunkUncleaned` | `from src.retrieval.models import InferenceChunkUncleaned` |
| `from onyx.db.enums import EmbeddingPrecision` | `from src.retrieval.models import EmbeddingPrecision` |
| `from onyx.document_index.interfaces_new import *` | `from src.backend.document_index.interfaces_new import *` |
| `from onyx.document_index.chunk_content_enrichment import *` | `from src.backend.document_index.chunk_content_enrichment import *` |
| `from onyx.document_index.opensearch.constants import *` | `from src.backend.document_index.opensearch.constants import *` |
| `from onyx.document_index.opensearch.schema import *` | `from src.backend.document_index.opensearch.schema import *` |
| `from onyx.document_index.opensearch.search import *` | `from src.backend.document_index.opensearch.search import *` |
| `from onyx.document_index.opensearch.client import *` | `from src.backend.document_index.opensearch.client import *` |
| `from onyx.indexing.models import DocMetadataAwareIndexChunk` | `from src.retrieval.models import DocMetadataAwareIndexChunk` |
| `from onyx.utils.logger import setup_logger` | `from src.backend.document_index.utils import setup_logger` |
| `from onyx.utils.text_processing import remove_invalid_unicode_chars` | `from src.backend.document_index.utils import remove_invalid_unicode_chars` |
| `from onyx.utils.batching import batch_generator` | `from src.backend.document_index.utils import batch_generator` |
| `from onyx.connectors.models import convert_metadata_list_of_strings_to_dict` | `from src.backend.document_index.utils import convert_metadata_list_of_strings_to_dict` |
| `from onyx.connectors.cross_connector_utils.miscellaneous_utils import get_experts_stores_representations` | `from src.backend.document_index.utils import get_experts_stores_representations` |
| `from onyx.configs.constants import PUBLIC_DOC_PAT` | `from src.backend.configs.constants import PUBLIC_DOC_PAT` |
| `from onyx.configs.constants import OnyxRedisLocks` | (delete — replace usages with string literals or stubs) |
| `from onyx.configs.app_configs import MAX_CHUNKS_PER_DOC_BATCH` | `MAX_CHUNKS_PER_DOC_BATCH = int(os.environ.get("MAX_CHUNKS_PER_DOC_BATCH", "512"))` |
| `from onyx.configs.app_configs import VERIFY_CREATE_OPENSEARCH_INDEX_ON_INIT_MT` | `VERIFY_CREATE_OPENSEARCH_INDEX_ON_INIT_MT = False` |
| `from onyx.db.models import DocumentSource` | `DocumentSource = str  # stub` |
| `from onyx.document_index.document_index_utils import *` | `from src.backend.document_index.document_index_utils import *` |
| `from onyx.redis.lock_context import redis_shared_lock` | (stub — see below) |
| `from onyx.key_value_store.factory import get_shared_kv_store` | (stub — see below) |
| `from shared_configs.model_server_models import Embedding` | `from src.retrieval.models import Embedding` |
| `from shared_configs.configs import MULTI_TENANT` | `MULTI_TENANT = os.environ.get("MULTI_TENANT", "").lower() in {"1","true","yes"}` |
| `from onyx.context.search.enums import QueryType` | `from src.retrieval.models import QueryType` |
| `from onyx.context.search.models import IndexFilters` | `from src.retrieval.models import IndexFilters` |

For `redis_shared_lock` and `get_shared_kv_store`, add this stub in `utils.py` and import from there:

Add to `src/backend/document_index/utils.py`:

```python
import contextlib

@contextlib.contextmanager
def redis_shared_lock(lock_name: str, *, ttl: int = 60, timeout: int = 60):
    """Stub lock context manager. Does not acquire a real Redis lock."""
    yield


class _NullKVStore:
    """Stub key-value store that always returns None."""
    def get(self, key: str) -> None:
        return None

    def set(self, key: str, value: object) -> None:
        pass

    def delete(self, key: str) -> None:
        pass


def get_shared_kv_store() -> _NullKVStore:
    """Stub — returns a no-op KV store. Replace with real Redis KV when needed."""
    return _NullKVStore()
```

Apply the full substitution pass to `opensearch_document_index.py`, `schema.py`, `search.py`, and `client.py`. Each file: find every `from onyx.*` or `from shared_configs.*` line and replace using the table above. Add `import os` at the top if not present.

- [ ] **Step 4: Run import tests**

```bash
pytest tests/unit/document_index/test_imports.py -v
```

Expected: OpenSearch tests PASS (others may still fail — they'll be fixed in Task 8)

- [ ] **Step 5: Commit**

```bash
git add src/backend/document_index/opensearch/ src/backend/document_index/utils.py tests/unit/document_index/test_imports.py
git commit -m "fix(document_index): replace onyx imports in opensearch implementation"
```

---

## Task 8: Fix Vespa implementation imports

**Files:**
- Modify: `src/backend/document_index/vespa/vespa_document_index.py`
- Modify: `src/backend/document_index/vespa/chunk_retrieval.py`
- Modify: `src/backend/document_index/vespa/deletion.py`
- Modify: `src/backend/document_index/vespa/indexing_utils.py`
- Modify: `src/backend/document_index/vespa/kg_interactions.py`
- Modify: `src/backend/document_index/vespa/shared_utils/utils.py`
- Modify: `src/backend/document_index/vespa/shared_utils/vespa_request_builders.py`
- Test: `tests/unit/document_index/test_imports.py` (extend)

- [ ] **Step 1: Add failing import tests**

Add to `tests/unit/document_index/test_imports.py`:

```python
def test_vespa_internal_types_importable():
    import src.backend.document_index.vespa.internal_types  # noqa: F401


def test_vespa_shared_utils_importable():
    import src.backend.document_index.vespa.shared_utils.utils  # noqa: F401
    import src.backend.document_index.vespa.shared_utils.vespa_request_builders  # noqa: F401


def test_vespa_chunk_retrieval_importable():
    import src.backend.document_index.vespa.chunk_retrieval  # noqa: F401


def test_vespa_deletion_importable():
    import src.backend.document_index.vespa.deletion  # noqa: F401


def test_vespa_indexing_utils_importable():
    import src.backend.document_index.vespa.indexing_utils  # noqa: F401


def test_vespa_document_index_importable():
    import src.backend.document_index.vespa.vespa_document_index  # noqa: F401
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/document_index/test_imports.py::test_vespa_chunk_retrieval_importable -v
```

Expected: FAIL

- [ ] **Step 3: Fix `vespa/shared_utils/utils.py` and `vespa_request_builders.py`**

Apply the import substitution table from Task 7. Additionally:

For any `from onyx.configs.constants import INDEX_SEPARATOR`:
```python
from src.backend.configs.constants import INDEX_SEPARATOR
```

For any `from onyx.kg.utils.formatting_utils import split_relationship_id`:
```python
from src.backend.document_index.utils import split_relationship_id
```

For `from onyx.background.celery.tasks.opensearch_migration.constants import ...` (in `chunk_retrieval.py`):
```python
# Stub: OpenSearch migration transformer not needed for Vespa-only paths
def transform_for_opensearch_migration(*args, **kwargs):
    raise NotImplementedError("OpenSearch migration transformer not implemented")

OPENSEARCH_MIGRATION_CHUNK_BATCH_SIZE = 100
```

For any remaining `from onyx.configs.app_configs import LOG_VESPA_TIMING_INFORMATION`:
```python
LOG_VESPA_TIMING_INFORMATION = False
```

For `from onyx.configs.app_configs import VESPA_MIGRATION_REQUEST_TIMEOUT_S`:
```python
VESPA_MIGRATION_REQUEST_TIMEOUT_S = int(os.environ.get("VESPA_MIGRATION_REQUEST_TIMEOUT_S", "30"))
```

Apply the same `from onyx.*` → local substitutions to all remaining vespa files: `deletion.py`, `indexing_utils.py`, `chunk_retrieval.py`, `kg_interactions.py`, `vespa_document_index.py`.

For `vespa_document_index.py` extra replacements:

| Old | New |
|---|---|
| `from onyx.key_value_store.factory import get_shared_kv_store` | `from src.backend.document_index.utils import get_shared_kv_store` |
| `from onyx.redis.lock_context import redis_shared_lock` | `from src.backend.document_index.utils import redis_shared_lock` |
| `from onyx.configs.chat_configs import DOC_TIME_DECAY` | `DOC_TIME_DECAY = float(os.environ.get("DOC_TIME_DECAY", "0.5"))` |
| `from onyx.configs.chat_configs import HYBRID_ALPHA` | `HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))` |
| `from onyx.configs.chat_configs import TITLE_CONTENT_RATIO` | `TITLE_CONTENT_RATIO = float(os.environ.get("TITLE_CONTENT_RATIO", "0.1"))` |
| `from onyx.configs.chat_configs import VESPA_SEARCHER_THREADS` | `VESPA_SEARCHER_THREADS = int(os.environ.get("VESPA_SEARCHER_THREADS", "8"))` |
| `from onyx.configs.app_configs import RECENCY_BIAS_MULTIPLIER` | `RECENCY_BIAS_MULTIPLIER = float(os.environ.get("RECENCY_BIAS_MULTIPLIER", "1.0"))` |
| `from onyx.configs.app_configs import RERANK_COUNT` | `RERANK_COUNT = int(os.environ.get("RERANK_COUNT", "0"))` |
| `from onyx.configs.constants import KV_REINDEX_KEY` | `from src.backend.configs.constants import KV_REINDEX_KEY` |

- [ ] **Step 4: Run all import tests**

```bash
pytest tests/unit/document_index/test_imports.py -v
```

Expected: All PASS

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: All previously passing tests still PASS; new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backend/document_index/vespa/ tests/unit/document_index/test_imports.py
git commit -m "fix(document_index): replace onyx imports in vespa implementation"
```

---

## Task 9: Wire `DisabledDocumentIndex` test and final verification

**Files:**
- Test: `tests/unit/document_index/test_disabled.py`

- [ ] **Step 1: Write and run disabled index tests**

Create `tests/unit/document_index/test_disabled.py`:

```python
import pytest
from src.backend.document_index.disabled import DisabledDocumentIndex
from src.retrieval.models import EmbeddingPrecision


def make_disabled() -> DisabledDocumentIndex:
    return DisabledDocumentIndex()


def test_verify_is_noop():
    idx = make_disabled()
    # Should not raise
    idx.verify_and_create_index_if_necessary(
        embedding_dim=768,
        embedding_precision=EmbeddingPrecision.FLOAT,
    )


def test_index_raises():
    idx = make_disabled()
    with pytest.raises(RuntimeError, match="DISABLE_VECTOR_DB"):
        idx.index(chunks=[], indexing_metadata=None)


def test_delete_raises():
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.delete(document_id="doc1")


def test_update_raises():
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.update(update_requests=[])


def test_id_based_retrieval_raises():
    from src.retrieval.models import IndexFilters
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.id_based_retrieval(chunk_requests=[], filters=IndexFilters())


def test_hybrid_retrieval_raises():
    from src.retrieval.models import IndexFilters, QueryType
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.hybrid_retrieval(
            query="test",
            query_embedding=[0.1] * 768,
            final_keywords=None,
            query_type=QueryType.HYBRID,
            filters=IndexFilters(),
            num_to_retrieve=10,
        )


def test_keyword_retrieval_raises():
    from src.retrieval.models import IndexFilters
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.keyword_retrieval(query="test", filters=IndexFilters(), num_to_retrieve=5)


def test_semantic_retrieval_raises():
    from src.retrieval.models import IndexFilters
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.semantic_retrieval(
            query_embedding=[0.1] * 768,
            filters=IndexFilters(),
            num_to_retrieve=5,
        )


def test_random_retrieval_raises():
    from src.retrieval.models import IndexFilters
    idx = make_disabled()
    with pytest.raises(RuntimeError):
        idx.random_retrieval(filters=IndexFilters())
```

```bash
pytest tests/unit/document_index/test_disabled.py -v
```

Expected: All 9 PASS

- [ ] **Step 2: Run full regression check**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: All unit tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/document_index/test_disabled.py
git commit -m "test(document_index): add DisabledDocumentIndex behavior tests"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|---|---|
| Convert `from onyx.*` imports | Tasks 4-8 |
| Add `InferenceChunk`, `IndexFilters`, `QueryType`, `Embedding` | Task 1 |
| Add `PUBLIC_DOC_PAT`, `RETURN_SEPARATOR`, constants | Task 2 |
| Add `VectorDbSettings` env vars | Task 2 |
| Create utility stubs (`batch_generator`, etc.) | Task 3 |
| Fix `document_index_utils.py` (remove DB session) | Task 5 |
| Fix `chunk_content_enrichment.py` | Task 5 |
| Fix `factory.py` (config-based, no DB) | Task 6 |
| Fix OpenSearch implementation | Task 7 |
| Fix Vespa implementation | Task 8 |
| `DisabledDocumentIndex` works correctly | Task 9 |
| All files importable without onyx | Tasks 4-8 + `test_imports.py` |

### Type Consistency

- `InferenceChunk` defined in Task 1, used in Tasks 4, 7, 8, 9.
- `IndexFilters` defined in Task 1, used in `interfaces_new.py`, `disabled.py`, `chunk_retrieval.py`, `vespa_request_builders.py`.
- `Embedding = list[float]` defined in Task 1, used in `interfaces_new.py`, `disabled.py`, `opensearch_document_index.py`.
- `DocMetadataAwareIndexChunk.embedded_chunk.chunk.document_id` used in `document_index_utils.py` Task 5.
- `MultipassConfig` defined in Task 1, used in `document_index_utils.py` Task 5.
- `PUBLIC_DOC_PAT` defined in Task 2, used in `interfaces_new.py` Task 4.
- `VectorDbSettings` defined in Task 2, used in `factory.py` Task 6 (via env vars).
