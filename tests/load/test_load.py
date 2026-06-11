"""Load tests: exercise FastAPI endpoints under concurrent traffic.

All tests use FastAPI's in-process TestClient so no live server is needed.
Run with:
    pytest tests/load/ -v -s -m load

Latency metrics (p50, p95) are printed to stdout via -s.
Thresholds are intentionally generous for CI (in-process transport is fast).
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.internal.servers.app import create_base_app, create_search_app

pytestmark = pytest.mark.load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    """Return the p-th percentile of *data* (0–100)."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (k - lo) * (data_sorted[hi] - data_sorted[lo])


def _run_concurrent(
    client: TestClient,
    method: str,
    path: str,
    *,
    workers: int,
    total: int,
    json: Any = None,
) -> tuple[list[float], list[int]]:
    """
    Fire *total* requests with *workers* threads.

    Returns (latencies_seconds, error_status_codes).
    """
    latencies: list[float] = []
    errors: list[int] = []

    def _call() -> tuple[int, float]:
        t0 = time.perf_counter()
        if method.upper() == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=json)
        return resp.status_code, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_call) for _ in range(total)]
        for fut in as_completed(futures):
            status, lat = fut.result()
            latencies.append(lat)
            if status >= 400:
                errors.append(status)

    return latencies, errors


def _print_stats(label: str, latencies: list[float]) -> None:
    p50 = _percentile(latencies, 50) * 1_000
    p95 = _percentile(latencies, 95) * 1_000
    p99 = _percentile(latencies, 99) * 1_000
    mean = statistics.mean(latencies) * 1_000
    print(
        f"\n[{label}] n={len(latencies)}  mean={mean:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def health_client():
    return TestClient(create_base_app("Load Test – Health"))


@pytest.fixture(scope="module")
def search_client():
    engine = MagicMock()
    engine.batch_search.side_effect = lambda queries: [["result"] for _ in queries]
    return TestClient(create_search_app("Load Test – Search", engine))


# ---------------------------------------------------------------------------
# /health load
# ---------------------------------------------------------------------------


class TestHealthEndpointLoad:
    WORKERS = 20
    TOTAL = 200

    def test_zero_errors_under_concurrent_load(self, health_client):
        _, errors = _run_concurrent(
            health_client, "GET", "/health", workers=self.WORKERS, total=self.TOTAL
        )
        assert errors == [], f"Unexpected HTTP errors: {errors}"

    def test_all_requests_complete(self, health_client):
        latencies, _ = _run_concurrent(
            health_client, "GET", "/health", workers=self.WORKERS, total=self.TOTAL
        )
        assert len(latencies) == self.TOTAL

    def test_p95_under_500ms(self, health_client):
        latencies, _ = _run_concurrent(
            health_client, "GET", "/health", workers=self.WORKERS, total=self.TOTAL
        )
        _print_stats("health p95", latencies)
        assert _percentile(latencies, 95) < 0.5

    def test_p99_under_1s(self, health_client):
        latencies, _ = _run_concurrent(
            health_client, "GET", "/health", workers=self.WORKERS, total=self.TOTAL
        )
        _print_stats("health p99", latencies)
        assert _percentile(latencies, 99) < 1.0


# ---------------------------------------------------------------------------
# /retrieve load
# ---------------------------------------------------------------------------


class TestSearchEndpointLoad:
    WORKERS = 10
    TOTAL = 100
    PAYLOAD = {"queries": ["what is retrieval augmented generation"]}

    def test_zero_errors_under_concurrent_load(self, search_client):
        _, errors = _run_concurrent(
            search_client,
            "POST",
            "/retrieve",
            workers=self.WORKERS,
            total=self.TOTAL,
            json=self.PAYLOAD,
        )
        assert errors == [], f"Unexpected HTTP errors: {errors}"

    def test_all_requests_complete(self, search_client):
        latencies, _ = _run_concurrent(
            search_client,
            "POST",
            "/retrieve",
            workers=self.WORKERS,
            total=self.TOTAL,
            json=self.PAYLOAD,
        )
        assert len(latencies) == self.TOTAL

    def test_p95_under_1s(self, search_client):
        latencies, _ = _run_concurrent(
            search_client,
            "POST",
            "/retrieve",
            workers=self.WORKERS,
            total=self.TOTAL,
            json=self.PAYLOAD,
        )
        _print_stats("search p95", latencies)
        assert _percentile(latencies, 95) < 1.0

    def test_throughput_exceeds_10_rps(self, search_client):
        t0 = time.perf_counter()
        _, errors = _run_concurrent(
            search_client,
            "POST",
            "/retrieve",
            workers=self.WORKERS,
            total=self.TOTAL,
            json=self.PAYLOAD,
        )
        rps = self.TOTAL / (time.perf_counter() - t0)
        print(f"\n[search throughput] {rps:.1f} req/s")
        assert errors == []
        assert rps > 10, f"Throughput {rps:.1f} req/s < 10 req/s"


# ---------------------------------------------------------------------------
# Batch-size scaling
# ---------------------------------------------------------------------------


class TestBatchSizeScaling:
    """Verify the endpoint handles varying query-batch sizes without errors."""

    WORKERS = 5
    TOTAL = 50

    @pytest.mark.parametrize("batch_size", [1, 5, 20])
    def test_varying_batch_sizes_produce_no_errors(self, batch_size):
        engine = MagicMock()
        engine.batch_search.side_effect = lambda queries: [["r"] for _ in queries]
        client = TestClient(create_search_app("Batch Scale", engine))

        payload = {"queries": [f"query_{i}" for i in range(batch_size)]}
        _, errors = _run_concurrent(
            client,
            "POST",
            "/retrieve",
            workers=self.WORKERS,
            total=self.TOTAL,
            json=payload,
        )
        assert errors == [], f"Errors with batch_size={batch_size}: {errors}"


# ---------------------------------------------------------------------------
# Mixed-endpoint concurrency
# ---------------------------------------------------------------------------


class TestMixedEndpointLoad:
    """Fire both /health and /retrieve concurrently to check for interference."""

    def test_mixed_requests_no_errors(self, health_client, search_client):
        payload = {"queries": ["mixed load query"]}

        def _health():
            return health_client.get("/health").status_code, 0.0

        def _search():
            t0 = time.perf_counter()
            status = search_client.post("/retrieve", json=payload).status_code
            return status, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = [pool.submit(_health) for _ in range(50)] + [
                pool.submit(_search) for _ in range(50)
            ]
            statuses = [f.result()[0] for f in as_completed(futs)]

        errors = [s for s in statuses if s >= 400]
        assert errors == [], f"Mixed load errors: {errors}"
