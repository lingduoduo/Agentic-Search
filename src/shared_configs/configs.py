"""Shared configuration constants for multi-tenant setup."""

import os

TENANT_ID_PREFIX = "tenant_"
POSTGRES_DEFAULT_SCHEMA = "public"

# Embedding model server connection
MODEL_SERVER_HOST: str = os.environ.get("MODEL_SERVER_HOST", "localhost")
MODEL_SERVER_PORT: int = int(os.environ.get("MODEL_SERVER_PORT", "9000"))

# Embedding context and batch sizes
DOC_EMBEDDING_CONTEXT_SIZE: int = int(
    os.environ.get("DOC_EMBEDDING_CONTEXT_SIZE", "512")
)
BATCH_SIZE_ENCODE_CHUNKS: int = int(os.environ.get("BATCH_SIZE_ENCODE_CHUNKS", "8"))
BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES: int = int(
    os.environ.get("BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES", "512")
)
VERTEXAI_EMBEDDING_LOCAL_BATCH_SIZE: int = int(
    os.environ.get("VERTEXAI_EMBEDDING_LOCAL_BATCH_SIZE", "5")
)

# Timeouts (seconds)
API_BASED_EMBEDDING_TIMEOUT: int = int(
    os.environ.get("API_BASED_EMBEDDING_TIMEOUT", "600")
)
OPENAI_EMBEDDING_TIMEOUT: int = int(os.environ.get("OPENAI_EMBEDDING_TIMEOUT", "600"))

# Runtime flags
INDEXING_ONLY: bool = os.environ.get("INDEXING_ONLY", "").lower() in {
    "1",
    "true",
    "yes",
}
SKIP_WARM_UP: bool = os.environ.get("SKIP_WARM_UP", "").lower() in {"1", "true", "yes"}

# Default encoder model (HuggingFace)
DOCUMENT_ENCODER_MODEL: str = os.environ.get(
    "DOCUMENT_ENCODER_MODEL", "nomic-ai/nomic-embed-text-v1"
)

# Indexing config
INDEXING_EMBEDDING_MODEL_NUM_THREADS: int = int(
    os.environ.get("INDEXING_EMBEDDING_MODEL_NUM_THREADS", "1")
)
LARGE_CHUNK_RATIO: int = int(os.environ.get("LARGE_CHUNK_RATIO", "4"))
