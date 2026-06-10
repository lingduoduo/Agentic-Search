"""Redis cache operations for hierarchy node ancestor resolution.

This module provides a Redis-based cache for hierarchy node parent relationships,
enabling fast ancestor path resolution without repeated database queries.

The cache stores node_id -> parent_id mappings for all hierarchy nodes of a given
source type. When resolving ancestors for a document, we walk up the tree using
Redis lookups instead of database queries.

Cache Strategy:
- Nodes are cached per source type with a 6-hour TTL
- During docfetching, nodes are added to cache as they're upserted to Postgres
- If the cache is stale (TTL expired during long-running job), one worker does
  a full refresh from DB while others wait
- If a node is still not found after refresh, we log an error and fall back to
  using only the SOURCE-type node as the ancestor
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import cast

from pydantic import BaseModel

from src.backend.configs.constants import DocumentSource
from src.backend.servers.redis.tenant_redis_client import TenantRedisClient

logger = logging.getLogger(__name__)

# Cache TTL: 6 hours in seconds
HIERARCHY_CACHE_TTL_SECONDS = 6 * 60 * 60

# Lock timeout for cache refresh: 5 minutes
HIERARCHY_CACHE_LOCK_TIMEOUT_SECONDS = 5 * 60

# Lock acquisition timeout: 60 seconds
HIERARCHY_CACHE_LOCK_ACQUIRE_TIMEOUT_SECONDS = 60

MAX_DEPTH = 1000


class HierarchyNodeType(str, Enum):
    SOURCE = "SOURCE"
    FOLDER = "FOLDER"
    FILE = "FILE"
    PAGE = "PAGE"
    SPACE = "SPACE"


class HierarchyNodeCacheEntry(BaseModel):
    """Represents a hierarchy node for caching purposes."""

    node_id: int
    parent_id: int | None
    node_type: HierarchyNodeType
    raw_node_id: str


def _cache_key(source: DocumentSource) -> str:
    """Get the Redis hash key for hierarchy node cache of a given source."""
    return f"hierarchy_cache:{source.value}"


def _raw_id_cache_key(source: DocumentSource) -> str:
    """Get the Redis hash key for raw_node_id -> node_id mapping."""
    return f"hierarchy_cache_rawid:{source.value}"


def _source_node_key(source: DocumentSource) -> str:
    """Get the Redis key for the SOURCE-type node ID of a given source."""
    return f"hierarchy_source_node:{source.value}"


def _loading_lock_key(source: DocumentSource) -> str:
    """Get the Redis lock key for cache loading of a given source."""
    return f"hierarchy_cache_loading:{source.value}"


def _construct_parent_value(parent_id: int | None, node_type: HierarchyNodeType) -> str:
    """Format: "parent_id:node_type" where parent_id is empty string if None."""
    parent_str = str(parent_id) if parent_id is not None else ""
    return f"{parent_str}:{node_type.value}"


def _unpack_parent_value(value: str) -> tuple[int | None, HierarchyNodeType | None]:
    """Unpack a cached value string back into (parent_id, node_type)."""
    parts = value.split(":", 1)
    parent_str = parts[0]
    node_type_str = parts[1] if len(parts) > 1 else ""
    parent_id = int(parent_str) if parent_str else None
    node_type = HierarchyNodeType(node_type_str) if node_type_str else None
    return parent_id, node_type


def cache_hierarchy_node(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    entry: HierarchyNodeCacheEntry,
) -> None:
    """Add or update a single hierarchy node in the Redis cache."""
    cache_key = _cache_key(source)
    raw_id_key = _raw_id_cache_key(source)

    value = _construct_parent_value(entry.parent_id, entry.node_type)
    redis_client.hset(cache_key, str(entry.node_id), value)
    redis_client.hset(raw_id_key, entry.raw_node_id, str(entry.node_id))

    if entry.node_type == HierarchyNodeType.SOURCE:
        source_node_key = _source_node_key(source)
        redis_client.set(source_node_key, str(entry.node_id))
        redis_client.expire(source_node_key, HIERARCHY_CACHE_TTL_SECONDS)

    redis_client.expire(cache_key, HIERARCHY_CACHE_TTL_SECONDS)
    redis_client.expire(raw_id_key, HIERARCHY_CACHE_TTL_SECONDS)


def cache_hierarchy_nodes_batch(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    entries: list[HierarchyNodeCacheEntry],
) -> None:
    """Add or update multiple hierarchy nodes in the Redis cache."""
    if not entries:
        return

    cache_key = _cache_key(source)
    raw_id_key = _raw_id_cache_key(source)
    source_node_key = _source_node_key(source)

    parent_mapping: dict[str, str] = {}
    raw_id_mapping: dict[str, str] = {}
    source_node_id: int | None = None

    for entry in entries:
        parent_mapping[str(entry.node_id)] = _construct_parent_value(
            entry.parent_id, entry.node_type
        )
        raw_id_mapping[entry.raw_node_id] = str(entry.node_id)
        if entry.node_type == HierarchyNodeType.SOURCE:
            source_node_id = entry.node_id

    redis_client.hset(cache_key, mapping=parent_mapping)
    redis_client.hset(raw_id_key, mapping=raw_id_mapping)

    if source_node_id is not None:
        redis_client.set(source_node_key, str(source_node_id))
        redis_client.expire(source_node_key, HIERARCHY_CACHE_TTL_SECONDS)

    redis_client.expire(cache_key, HIERARCHY_CACHE_TTL_SECONDS)
    redis_client.expire(raw_id_key, HIERARCHY_CACHE_TTL_SECONDS)


def evict_hierarchy_nodes_from_cache(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    raw_node_ids: list[str],
) -> None:
    """Remove specific hierarchy nodes from the Redis cache."""
    if not raw_node_ids:
        return

    cache_key = _cache_key(source)
    raw_id_key = _raw_id_cache_key(source)

    raw_values = cast(list[str | None], redis_client.hmget(raw_id_key, raw_node_ids))
    node_id_strs = [v for v in raw_values if v is not None]

    if node_id_strs:
        redis_client.hdel(cache_key, *node_id_strs)
    redis_client.hdel(raw_id_key, *raw_node_ids)


def get_node_id_from_raw_id(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    raw_node_id: str,
) -> tuple[int | None, bool]:
    """Get the database node_id for a raw_node_id from the cache.

    Returns (node_id, found_in_cache).
    """
    raw_id_key = _raw_id_cache_key(source)
    value = redis_client.hget(raw_id_key, raw_node_id)

    if value is None:
        return None, False

    value_str: str
    if isinstance(value, bytes):
        value_str = value.decode("utf-8")
    else:
        value_str = str(value)

    return int(value_str), True


def get_parent_id_from_cache(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    node_id: int,
) -> tuple[int | None, bool]:
    """Get the parent_id for a node from the cache.

    Returns (parent_id, found_in_cache).
    """
    cache_key = _cache_key(source)
    value = redis_client.hget(cache_key, str(node_id))

    if value is None:
        return None, False

    value_str: str
    if isinstance(value, bytes):
        value_str = value.decode("utf-8")
    else:
        value_str = str(value)

    parent_id, _ = _unpack_parent_value(value_str)
    return parent_id, True


def is_cache_populated(redis_client: TenantRedisClient, source: DocumentSource) -> bool:
    """Check if the cache has any entries for this source."""
    cache_key = _cache_key(source)
    return redis_client.exists(cache_key) > 0


def clear_hierarchy_cache(
    redis_client: TenantRedisClient, source: DocumentSource
) -> None:
    """Clear the hierarchy cache for a source (useful for testing)."""
    cache_key = _cache_key(source)
    raw_id_key = _raw_id_cache_key(source)
    source_node_key = _source_node_key(source)
    redis_client.delete(cache_key)
    redis_client.delete(raw_id_key)
    redis_client.delete(source_node_key)


# ---------------------------------------------------------------------------
# DB-dependent functions — stubbed; require SQLAlchemy session from onyx.db.*
# ---------------------------------------------------------------------------


def refresh_hierarchy_cache_from_db(
    redis_client: TenantRedisClient,
    db_session: object,
    source: DocumentSource,
) -> None:
    raise NotImplementedError(
        "refresh_hierarchy_cache_from_db requires onyx.db — not available in this repo"
    )


def _walk_ancestor_chain(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    start_node_id: int,
    db_session: object,
) -> list[int]:
    raise NotImplementedError(
        "_walk_ancestor_chain requires onyx.db — not available in this repo"
    )


def get_ancestors_from_raw_id(
    redis_client: TenantRedisClient,
    source: DocumentSource,
    parent_hierarchy_raw_node_id: str | None,
    db_session: object,
) -> list[int]:
    raise NotImplementedError(
        "get_ancestors_from_raw_id requires onyx.db — not available in this repo"
    )


def get_source_node_id_from_cache(
    redis_client: TenantRedisClient,
    db_session: object,
    source: DocumentSource,
) -> int | None:
    raise NotImplementedError(
        "get_source_node_id_from_cache requires onyx.db — not available in this repo"
    )


def ensure_source_node_exists(
    redis_client: TenantRedisClient,
    db_session: object,
    source: DocumentSource,
) -> int:
    raise NotImplementedError(
        "ensure_source_node_exists requires onyx.db — not available in this repo"
    )
