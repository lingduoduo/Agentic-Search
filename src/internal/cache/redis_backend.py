from redis import Redis
from redis.lock import Lock as RedisLock

from src.internal.cache.interface import CacheBackend
from src.internal.cache.interface import CacheLock


class RedisCacheLock(CacheLock):
    """Wraps ``redis.lock.Lock`` behind the ``CacheLock`` interface."""

    def __init__(self, lock: RedisLock) -> None:
        self._lock = lock

    def acquire(
        self,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> bool:
        return bool(
            self._lock.acquire(
                blocking=blocking,
                blocking_timeout=blocking_timeout,
            )
        )

    def release(self) -> None:
        self._lock.release()

    def owned(self) -> bool:
        return bool(self._lock.owned())


class RedisCacheBackend(CacheBackend):
    """``CacheBackend`` implementation that delegates to a Redis client.

    Key-prefixing for tenant isolation must be handled by the caller when
    constructing the client (e.g. via a tenant-aware Redis wrapper).
    """

    def __init__(self, redis_client: Redis) -> None:
        self._r = redis_client

    def get(self, key: str) -> bytes | None:
        return self._r.get(key)

    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> None:
        self._r.set(key, value, ex=ex)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._r.exists(key))

    def expire(self, key: str, seconds: int) -> None:
        self._r.expire(key, seconds)

    def ttl(self, key: str) -> int:
        return self._r.ttl(key)

    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        return RedisCacheLock(self._r.lock(name, timeout=timeout, thread_local=False))

    def rpush(self, key: str, value: str | bytes) -> None:
        self._r.rpush(key, value)

    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        return self._r.blpop(keys, timeout=timeout)
