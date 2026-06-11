import os

MINI_CHUNK_SIZE: int = int(os.environ.get("MINI_CHUNK_SIZE", "256"))
SKIP_METADATA_IN_CHUNK: bool = (
    os.environ.get("SKIP_METADATA_IN_CHUNK", "false").lower() == "true"
)
USE_CHUNK_SUMMARY: bool = os.environ.get("USE_CHUNK_SUMMARY", "false").lower() == "true"
USE_DOCUMENT_SUMMARY: bool = (
    os.environ.get("USE_DOCUMENT_SUMMARY", "false").lower() == "true"
)
AVERAGE_SUMMARY_EMBEDDINGS: bool = (
    os.environ.get("AVERAGE_SUMMARY_EMBEDDINGS", "false").lower() == "true"
)
ENABLE_CONTEXTUAL_RAG: bool = (
    os.environ.get("ENABLE_CONTEXTUAL_RAG", "false").lower() == "true"
)
MAX_CHUNKS_PER_DOC_BATCH: int = int(os.environ.get("MAX_CHUNKS_PER_DOC_BATCH", "512"))
MAX_DOCUMENT_CHARS: int = int(os.environ.get("MAX_DOCUMENT_CHARS", "100000"))
MAX_TOKENS_FOR_FULL_INCLUSION: int = int(
    os.environ.get("MAX_TOKENS_FOR_FULL_INCLUSION", "512")
)
MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", "4096"))
STRICT_CHUNK_TOKEN_LIMIT: bool = (
    os.environ.get("STRICT_CHUNK_TOKEN_LIMIT", "false").lower() == "true"
)

# Metadata keys ignored during indexing (connector-internal fields)
_METADATA_KEYS_TO_IGNORE: frozenset[str] = frozenset(
    {
        "connector_id",
        "credential_id",
        "doc_updated_at",
        "primary_owners",
        "secondary_owners",
        "from_ingestion_api",
    }
)


def get_metadata_keys_to_ignore() -> frozenset[str]:
    return _METADATA_KEYS_TO_IGNORE
