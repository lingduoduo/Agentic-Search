from pathlib import Path


_TOP_LEVEL_COLLECTION_IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".understand-anything",
    "agentic_search.egg-info",
    "data",
    "dist",
    "indexes",
    "models",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    try:
        relative = collection_path.relative_to(config.rootpath)
    except ValueError:
        return False
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in _TOP_LEVEL_COLLECTION_IGNORE_DIRS:
        return True
    return len(parts) >= 2 and parts[:2] in {
        ("web", "dist"),
        ("web", "node_modules"),
    }


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "load: marks tests as load/performance tests")
    config.addinivalue_line(
        "markers", "integration: marks tests requiring a live server stack"
    )
    config.addinivalue_line(
        "markers", "alembic: marks migration tests that exercise Alembic flows"
    )
