"""Compatibility exports for indexing data models.

The canonical model definitions live in :mod:`src.retrieval.models`. This
module keeps the sampled ``src.servers.indexing`` layout while avoiding a second
copy of the same indexing contracts.
"""

from src.retrieval.models import BaseChunk
from src.retrieval.models import ChunkEmbedding
from src.retrieval.models import ChunkingConfig
from src.retrieval.models import DocMetadataAwareIndexChunk
from src.retrieval.models import DocumentAccess
from src.retrieval.models import EmbeddedChunk
from src.retrieval.models import EmbeddingConfig
from src.retrieval.models import EmbeddingModelDetail
from src.retrieval.models import EmbeddingPrecision
from src.retrieval.models import IndexChunk
from src.retrieval.models import IndexingPipelineConfig
from src.retrieval.models import IndexingPipelineResult
from src.retrieval.models import IndexingSetting
from src.retrieval.models import IndexWriterConfig
from src.retrieval.models import SwitchoverType

__all__ = [
    "BaseChunk",
    "ChunkEmbedding",
    "ChunkingConfig",
    "DocMetadataAwareIndexChunk",
    "DocumentAccess",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "EmbeddingModelDetail",
    "EmbeddingPrecision",
    "IndexChunk",
    "IndexingPipelineConfig",
    "IndexingPipelineResult",
    "IndexingSetting",
    "IndexWriterConfig",
    "SwitchoverType",
]
