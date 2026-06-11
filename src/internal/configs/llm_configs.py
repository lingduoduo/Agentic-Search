DEFAULT_IMAGE_ANALYSIS_MAX_SIZE_MB = 20


def get_image_extraction_and_analysis_enabled() -> bool:
    """Return the workspace setting for image extraction/analysis.

    Stubs out the DB/Redis-backed settings store — returns False so vision-LLM
    code paths are disabled by default in environments without a settings store.
    """
    try:
        from src.internal.servers.web.settings_store import load_settings

        settings = load_settings()
        if settings.image_extraction_and_analysis_enabled is not None:
            return settings.image_extraction_and_analysis_enabled
    except Exception:
        pass

    return False


def get_image_analysis_max_size_mb() -> int:
    """Get image analysis max size MB from workspace settings or env fallback."""
    try:
        from src.internal.servers.web.settings_store import load_settings

        settings = load_settings()
        if settings.image_analysis_max_size_mb is not None:
            return settings.image_analysis_max_size_mb
    except Exception:
        pass

    return DEFAULT_IMAGE_ANALYSIS_MAX_SIZE_MB
