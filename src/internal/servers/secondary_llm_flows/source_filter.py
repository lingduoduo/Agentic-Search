from __future__ import annotations

import logging

from src.internal.configs.constants import DocumentSource
from src.internal.llm.interfaces import LLM

logger = logging.getLogger(__name__)


def strings_to_document_sources(source_strs: list[str]) -> list[DocumentSource]:
    sources = []
    for s in source_strs:
        try:
            sources.append(DocumentSource(s))
        except ValueError:
            logger.warning("Failed to translate %s to a DocumentSource", s)
    return sources


def extract_source_filter(query: str, llm: LLM) -> list[DocumentSource] | None:
    raise NotImplementedError("This function should not be getting called right now")
