import json
import logging
import os

logger = logging.getLogger(__name__)

#####
# Embedding/Reranking Model Configs
#####
DEFAULT_DOCUMENT_ENCODER_MODEL = "nomic-ai/nomic-embed-text-v1"
DOCUMENT_ENCODER_MODEL = (
    os.environ.get("DOCUMENT_ENCODER_MODEL") or DEFAULT_DOCUMENT_ENCODER_MODEL
)
DOC_EMBEDDING_DIM = int(os.environ.get("DOC_EMBEDDING_DIM") or 768)
NORMALIZE_EMBEDDINGS = (
    os.environ.get("NORMALIZE_EMBEDDINGS") or "true"
).lower() == "true"

OLD_DEFAULT_DOCUMENT_ENCODER_MODEL = "thenlper/gte-small"
OLD_DEFAULT_MODEL_DOC_EMBEDDING_DIM = 384
OLD_DEFAULT_MODEL_NORMALIZE_EMBEDDINGS = False

SIM_SCORE_RANGE_LOW = float(os.environ.get("SIM_SCORE_RANGE_LOW") or 0.0)
SIM_SCORE_RANGE_HIGH = float(os.environ.get("SIM_SCORE_RANGE_HIGH") or 1.0)
ASYM_QUERY_PREFIX = os.environ.get("ASYM_QUERY_PREFIX", "search_query: ")
ASYM_PASSAGE_PREFIX = os.environ.get("ASYM_PASSAGE_PREFIX", "search_document: ")

EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE") or 0) or None

BATCH_SIZE_ENCODE_CHUNKS = EMBEDDING_BATCH_SIZE or 8
BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES = EMBEDDING_BATCH_SIZE or 512
CROSS_ENCODER_RANGE_MAX = 1
CROSS_ENCODER_RANGE_MIN = 0

#####
# Generative AI Model Configs
#####
GEN_AI_API_KEY = os.environ.get("GEN_AI_API_KEY")
GEN_AI_MODEL_VERSION = os.environ.get("GEN_AI_MODEL_VERSION")

GEN_AI_MAX_TOKENS = int(os.environ.get("GEN_AI_MAX_TOKENS") or 0) or None

GEN_AI_NUM_RESERVED_OUTPUT_TOKENS = int(
    os.environ.get("GEN_AI_NUM_RESERVED_OUTPUT_TOKENS") or 1024
)

GEN_AI_MODEL_FALLBACK_MAX_TOKENS = int(
    os.environ.get("GEN_AI_MODEL_FALLBACK_MAX_TOKENS") or 32000
)

GEN_AI_SINGLE_USER_MESSAGE_EXPECTED_MAX_TOKENS = 512
GEN_AI_TEMPERATURE = float(os.environ.get("GEN_AI_TEMPERATURE") or 0)

DISABLE_LITELLM_STREAMING = (
    os.environ.get("DISABLE_LITELLM_STREAMING") or "false"
).lower() == "true"

LITELLM_EXTRA_HEADERS: dict[str, str] | None = None
_LITELLM_EXTRA_HEADERS_RAW = os.environ.get("LITELLM_EXTRA_HEADERS")
if _LITELLM_EXTRA_HEADERS_RAW:
    try:
        LITELLM_EXTRA_HEADERS = json.loads(_LITELLM_EXTRA_HEADERS_RAW)
    except Exception:
        logger.error(
            "Failed to parse LITELLM_EXTRA_HEADERS, must be a valid JSON object"
        )

LITELLM_PASS_THROUGH_HEADERS: list[str] | None = None
_LITELLM_PASS_THROUGH_HEADERS_RAW = os.environ.get("LITELLM_PASS_THROUGH_HEADERS")
if _LITELLM_PASS_THROUGH_HEADERS_RAW:
    try:
        LITELLM_PASS_THROUGH_HEADERS = json.loads(_LITELLM_PASS_THROUGH_HEADERS_RAW)
    except Exception:
        logger.error(
            "Failed to parse LITELLM_PASS_THROUGH_HEADERS, must be a valid JSON object"
        )

LITELLM_EXTRA_BODY: dict | None = None
_LITELLM_EXTRA_BODY_RAW = os.environ.get("LITELLM_EXTRA_BODY")
if _LITELLM_EXTRA_BODY_RAW:
    try:
        LITELLM_EXTRA_BODY = json.loads(_LITELLM_EXTRA_BODY_RAW)
    except Exception:
        pass

#####
# Prompt Caching Configs
#####
ENABLE_PROMPT_CACHING = (
    os.environ.get("ENABLE_PROMPT_CACHING", "true").lower() != "false"
)

PROMPT_CACHE_REDIS_TTL_MULTIPLIER = float(
    os.environ.get("PROMPT_CACHE_REDIS_TTL_MULTIPLIER") or 1.2
)
