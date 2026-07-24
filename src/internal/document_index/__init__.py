"""Document-index backends, text handling, and indexing entry points."""

_BUILDER_EXPORTS = {
    "IndexBuilder",
    "IndexBuilderConfig",
    "IndexingHeartbeatInterface",
}


def __getattr__(name: str):
    if name in _BUILDER_EXPORTS:
        from . import index_builder

        return getattr(index_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
