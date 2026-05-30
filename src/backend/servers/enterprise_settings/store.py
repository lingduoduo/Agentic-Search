"""File-backed persistence for enterprise settings, logo, and analytics script.

The sampled Onyx version used get_kv_store() (Redis/DB-backed) and
get_default_file_store() (S3/local blob store). This repo uses plain
filesystem files inside AGENTIC_SEARCH_DATA_DIR so no external service
is required.

Directory layout:
    $AGENTIC_SEARCH_DATA_DIR/
        enterprise_settings.json   ← EnterpriseSettings JSON
        logo.(png|jpg)             ← custom logo binary
        logotype.(png|jpg)         ← custom logotype binary
        analytics_script.js        ← custom analytics script text
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from src.backend.servers.enterprise_settings.models import AnalyticsScriptUpload
from src.backend.servers.enterprise_settings.models import EnterpriseSettings

logger = logging.getLogger(__name__)

_DEFAULT_APP_NAME = "Agentic Search"
_CUSTOM_ANALYTICS_SECRET_KEY_ENV = "CUSTOM_ANALYTICS_SECRET_KEY"


def _data_dir() -> Path:
    path = Path(
        os.environ.get(
            "AGENTIC_SEARCH_DATA_DIR",
            Path.home() / ".local/share/agentic_search",
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path() -> Path:
    return _data_dir() / "enterprise_settings.json"


def _analytics_script_path() -> Path:
    return _data_dir() / "analytics_script.js"


def _logo_path() -> Path:
    return _data_dir() / "logo.bin"


def _logotype_path() -> Path:
    return _data_dir() / "logotype.bin"


def _logo_meta_path() -> Path:
    return _data_dir() / "logo_meta.json"


def _logotype_meta_path() -> Path:
    return _data_dir() / "logotype_meta.json"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def load_settings() -> EnterpriseSettings:
    """Load enterprise settings from disk, returning defaults if absent."""
    path = _settings_path()
    try:
        return EnterpriseSettings.model_validate_json(path.read_text())
    except (FileNotFoundError, ValueError):
        return EnterpriseSettings()


def store_settings(settings: EnterpriseSettings) -> None:
    """Persist *settings* to disk."""
    _settings_path().write_text(settings.model_dump_json())


def load_runtime_settings() -> EnterpriseSettings:
    """Like load_settings() but applies runtime defaults (not stored back)."""
    settings = load_settings()
    if not settings.application_name:
        settings = settings.model_copy(update={"application_name": _DEFAULT_APP_NAME})
    return settings


# ---------------------------------------------------------------------------
# Logo / logotype
# ---------------------------------------------------------------------------


def _valid_image(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def _guess_mime(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


def upload_logo(file: UploadFile | str, *, is_logotype: bool = False) -> bool:
    """Save the logo binary and its MIME type to disk. Returns False on error."""
    data_path = _logotype_path() if is_logotype else _logo_path()
    meta_path = _logotype_meta_path() if is_logotype else _logo_meta_path()

    if isinstance(file, str):
        if not os.path.isfile(file) or not _valid_image(file):
            logger.error("Invalid logo path or type: %s", file)
            return False
        with open(file, "rb") as fh:
            raw = fh.read()
        mime = _guess_mime(file)
    else:
        if not file.filename or not _valid_image(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type — only .png, .jpg, .jpeg are allowed.",
            )
        raw = file.file.read()
        mime = file.content_type or _guess_mime(file.filename)

    data_path.write_bytes(raw)
    meta_path.write_text(json.dumps({"mime": mime}))
    return True


def get_logo_bytes(*, is_logotype: bool = False) -> tuple[bytes, str] | None:
    """Return ``(raw_bytes, mime_type)`` or ``None`` if no logo is stored."""
    data_path = _logotype_path() if is_logotype else _logo_path()
    meta_path = _logotype_meta_path() if is_logotype else _logo_meta_path()
    if not data_path.exists():
        return None
    raw = data_path.read_bytes()
    mime = "image/png"
    try:
        mime = json.loads(meta_path.read_text()).get("mime", mime)
    except (FileNotFoundError, ValueError):
        pass
    return raw, mime


# ---------------------------------------------------------------------------
# Analytics script
# ---------------------------------------------------------------------------


def load_analytics_script() -> str | None:
    path = _analytics_script_path()
    return path.read_text() if path.exists() else None


def store_analytics_script(upload: AnalyticsScriptUpload) -> None:
    expected = os.environ.get(_CUSTOM_ANALYTICS_SECRET_KEY_ENV)
    if not expected or upload.secret_key != expected:
        raise ValueError("Invalid secret key for analytics script upload.")
    _analytics_script_path().write_text(upload.script)


__all__ = [
    "get_logo_bytes",
    "load_analytics_script",
    "load_runtime_settings",
    "load_settings",
    "store_analytics_script",
    "store_settings",
    "upload_logo",
]
