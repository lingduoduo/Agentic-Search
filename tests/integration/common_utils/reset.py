"""Reset helpers for integration tests.

In this deployment tests run against a live FastAPI server backed by SQLite.
There is no Postgres or Vespa to reset — the server manages its own state.
reset_all() and reset_all_multitenant() are no-ops unless SKIP_RESET is
explicitly set, matching the original behaviour when that flag is set.
"""

import logging
import os

from tests.integration.common_utils.constants import API_SERVER_URL

logger = logging.getLogger(__name__)


def _seed_dev_license_if_set() -> None:
    """Seed a dev license via the /license/upload endpoint if AGENTIC_SEARCH_DEV_LICENSE is set."""
    blob = os.environ.get("AGENTIC_SEARCH_DEV_LICENSE", "").strip()
    if not blob:
        return

    _PEM_BEGIN = "-----BEGIN AGENTIC SEARCH LICENSE-----"
    _PEM_END = "-----END AGENTIC SEARCH LICENSE-----"
    if blob.startswith(_PEM_BEGIN) and blob.endswith(_PEM_END):
        blob = "\n".join(blob.split("\n")[1:-1]).strip()

    try:
        import requests

        resp = requests.post(
            f"{API_SERVER_URL}/license/upload",
            files={"license_file": ("license.lic", blob.encode(), "text/plain")},
            timeout=30,
        )
        if resp.ok:
            logger.info("Dev license seeded via /license/upload")
        else:
            logger.warning("Dev license upload returned %s", resp.status_code)
    except Exception as exc:
        logger.warning("Could not seed dev license: %s", exc)


def reset_all() -> None:
    """No-op reset for SQLite-backed integration tests.

    The server creates a fresh in-memory / file-based SQLite store at startup.
    Tests that need isolation should run against a dedicated server instance or
    use unique identifiers to avoid cross-test interference.

    Setting SKIP_RESET=true is honoured for consistency with the original API.
    """
    if os.environ.get("SKIP_RESET", "").lower() == "true":
        logger.info("Skipping reset (SKIP_RESET=true).")
        return

    logger.info("reset_all: no-op for SQLite-backed server")
    _seed_dev_license_if_set()


def reset_all_multitenant() -> None:
    """No-op reset for multitenant tests."""
    if os.environ.get("SKIP_RESET", "").lower() == "true":
        logger.info("SKIPPING multitenant reset due to SKIP_RESET=true")
        return

    logger.info("reset_all_multitenant: no-op for SQLite-backed server")
