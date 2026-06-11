from __future__ import annotations

import json
import logging
import re
from enum import Enum

from pydantic import BaseModel

from src.internal.document_index.models import InferenceChunk
from src.internal.llm.interfaces import LLM
from src.internal.llm.models import ReasoningEffort
from src.internal.llm.models import UserMessage
from src.internal.prompts.search_prompts import DOCUMENT_CONTEXT_SELECTION_PROMPT
from src.internal.prompts.search_prompts import DOCUMENT_SELECTION_PROMPT
from src.internal.prompts.search_prompts import MAX_CHUNKS_FOR_RELEVANCE
from src.internal.prompts.search_prompts import TRY_TO_FILL_TO_MAX_INSTRUCTIONS

logger = logging.getLogger(__name__)


class ContextExpansionType(str, Enum):
    NOT_RELEVANT = "not_relevant"
    MAIN_SECTION_ONLY = "main_section_only"
    INCLUDE_ADJACENT_SECTIONS = "include_adjacent_sections"
    FULL_DOCUMENT = "full_document"


class InferenceSection(BaseModel):
    """A group of chunks centred around one primary (center) chunk."""

    center_chunk: InferenceChunk
    chunks: list[InferenceChunk] = []

    @property
    def combined_content(self) -> str:
        if self.chunks:
            return " ".join(c.content for c in self.chunks)
        return self.center_chunk.content

    model_config = {"arbitrary_types_allowed": True}


def select_chunks_for_relevance(
    section: InferenceSection,
    max_chunks: int = MAX_CHUNKS_FOR_RELEVANCE,
) -> list[InferenceChunk]:
    """Select a subset of chunks from a section based on center chunk position."""
    if max_chunks <= 0:
        return []

    center_chunk = section.center_chunk
    all_chunks = section.chunks

    try:
        center_index = next(
            i
            for i, chunk in enumerate(all_chunks)
            if chunk.chunk_ind == center_chunk.chunk_ind
        )
    except StopIteration:
        return [center_chunk]

    if max_chunks == 1:
        return [center_chunk]

    chunks_needed = max_chunks - 1
    chunks_before_available = center_index
    chunks_after_available = len(all_chunks) - center_index - 1

    chunks_before = min(chunks_needed // 2, chunks_before_available)
    chunks_after = min(chunks_needed // 2, chunks_after_available)

    remaining = chunks_needed - chunks_before - chunks_after
    if remaining > 0:
        if chunks_before_available > chunks_before:
            additional_before = min(remaining, chunks_before_available - chunks_before)
            chunks_before += additional_before
            remaining -= additional_before
        if remaining > 0 and chunks_after_available > chunks_after:
            additional_after = min(remaining, chunks_after_available - chunks_after)
            chunks_after += additional_after

    start_index = center_index - chunks_before
    end_index = center_index + chunks_after + 1

    return all_chunks[start_index:end_index]


def classify_section_relevance(
    document_title: str,
    section_text: str,
    user_query: str,
    llm: LLM,
    section_above_text: str | None,
    section_below_text: str | None,
) -> ContextExpansionType:
    """Use LLM to classify section relevance and determine context expansion type."""
    prompt_text = DOCUMENT_CONTEXT_SELECTION_PROMPT.format(
        document_title=document_title,
        main_section=section_text,
        section_above=section_above_text if section_above_text else "N/A",
        section_below=section_below_text if section_below_text else "N/A",
        user_query=user_query,
    )

    default_classification = ContextExpansionType.MAIN_SECTION_ONLY

    try:
        prompt_msg = UserMessage(content=prompt_text)
        response = llm.invoke(prompt=prompt_msg, reasoning_effort=ReasoningEffort.OFF)
        llm_response = response.choice.message.content

        if not llm_response:
            logger.warning(
                "LLM returned empty response for context selection, defaulting to MAIN_SECTION_ONLY"
            )
            classification = default_classification
        else:
            numbers = re.findall(r"\b[0-3]\b", llm_response)
            if numbers:
                situation = int(numbers[-1])
                situation_to_type = {
                    0: ContextExpansionType.NOT_RELEVANT,
                    1: ContextExpansionType.MAIN_SECTION_ONLY,
                    2: ContextExpansionType.INCLUDE_ADJACENT_SECTIONS,
                    3: ContextExpansionType.FULL_DOCUMENT,
                }
                classification = situation_to_type.get(
                    situation, default_classification
                )
            else:
                logger.warning(
                    "Could not parse situation number from LLM response: %s",
                    llm_response,
                )
                classification = default_classification

    except Exception as e:
        logger.error("Error calling LLM for context selection: %s", e)
        classification = default_classification

    if (
        not section_above_text
        and not section_below_text
        and classification != ContextExpansionType.NOT_RELEVANT
    ):
        classification = ContextExpansionType.MAIN_SECTION_ONLY

    return classification


def select_sections_for_expansion(
    sections: list[InferenceSection],
    user_query: str,
    llm: LLM,
    max_sections: int = 10,
    max_chunks_per_section: int | None = MAX_CHUNKS_FOR_RELEVANCE,
    try_to_fill_to_max: bool = False,
) -> tuple[list[InferenceSection], list[str] | None]:
    """Use LLM to select the most relevant document sections for expansion."""
    if not sections:
        return [], None

    section_map: dict[str, InferenceSection] = {}
    sections_dict: list[dict[str, str | int | list[str]]] = []

    for idx, section in enumerate(sections):
        section_id = f"{idx}"
        section_map[section_id] = section
        chunk = section.center_chunk

        authors = None
        if getattr(chunk, "primary_owners", None) or getattr(
            chunk, "secondary_owners", None
        ):
            authors = []
            if getattr(chunk, "primary_owners", None):
                authors.extend(chunk.primary_owners)  # type: ignore[union-attr]
            if getattr(chunk, "secondary_owners", None):
                authors.extend(chunk.secondary_owners)  # type: ignore[union-attr]

        updated_at_str = None
        if getattr(chunk, "updated_at", None):
            updated_at_str = chunk.updated_at.isoformat()  # type: ignore[union-attr]

        metadata_str = json.dumps(chunk.metadata)

        if max_chunks_per_section is not None:
            selected_chunks = select_chunks_for_relevance(
                section, max_chunks_per_section
            )
            selected_content = " ".join(c.content for c in selected_chunks)
        else:
            selected_content = section.combined_content

        section_dict: dict[str, str | int | list[str]] = {
            "section_id": idx,
            "title": chunk.semantic_identifier,
        }
        if updated_at_str is not None:
            section_dict["updated_at"] = updated_at_str
        if authors is not None:
            section_dict["authors"] = authors
        section_dict["source_type"] = str(chunk.source_type)
        section_dict["metadata"] = metadata_str
        section_dict["content"] = selected_content

        sections_dict.append(section_dict)

    extra_instructions = TRY_TO_FILL_TO_MAX_INSTRUCTIONS if try_to_fill_to_max else ""
    prompt_text = UserMessage(
        content=DOCUMENT_SELECTION_PROMPT.format(
            max_sections=max_sections,
            extra_instructions=extra_instructions,
            formatted_doc_sections=json.dumps(sections_dict, indent=2),
            user_query=user_query,
        )
    )

    try:
        response = llm.invoke(
            prompt=[prompt_text], reasoning_effort=ReasoningEffort.OFF
        )
        llm_response = response.choice.message.content

        if not llm_response:
            logger.warning(
                "LLM returned empty response for document selection, returning first max_sections"
            )
            return sections[:max_sections], None

        section_ids = []
        sections_with_exclamation: set[str] = set()

        bracket_match = re.search(r"\[([^\]]+)\]", llm_response)
        if bracket_match:
            parts = [part.strip() for part in bracket_match.group(1).split(",")]
            for part in parts:
                has_exclamation = "!" in part
                numbers = re.findall(r"\d+", part)
                if numbers:
                    section_id = numbers[0]
                    section_ids.append(section_id)
                    if has_exclamation:
                        sections_with_exclamation.add(section_id)
        else:
            comma_match = re.search(r"\b\d+!?\b(?:\s*,\s*\b\d+!?\b)*", llm_response)
            if comma_match:
                parts = [part.strip() for part in comma_match.group(0).split(",")]
                for part in parts:
                    has_exclamation = "!" in part
                    numbers = re.findall(r"\d+", part)
                    if numbers:
                        section_id = numbers[0]
                        section_ids.append(section_id)
                        if has_exclamation:
                            sections_with_exclamation.add(section_id)
            else:
                for match in re.finditer(r"\b(\d+)(!)?\b", llm_response):
                    section_id = match.group(1)
                    section_ids.append(section_id)
                    if match.group(2) == "!":
                        sections_with_exclamation.add(section_id)

        if not section_ids:
            logger.warning(
                "Could not parse section IDs from LLM response: %s", llm_response
            )
            return sections[:max_sections], None

        selected_sections = []
        document_ids_with_exclamation: list[str] = []
        num_sections = len(sections)

        for section_id_str in section_ids:
            try:
                section_id_int = int(section_id_str)
            except ValueError:
                logger.warning(
                    "Could not convert section ID to int: %s", section_id_str
                )
                continue

            if section_id_int < 0 or section_id_int >= num_sections:
                logger.warning(
                    "Section ID %s is out of range [0, %s], skipping",
                    section_id_int,
                    num_sections - 1,
                )
                continue

            section_id = str(section_id_int)
            if section_id in section_map:
                section = section_map[section_id]
                selected_sections.append(section)
                if section_id_str in sections_with_exclamation:
                    document_id = section.center_chunk.document_id
                    if document_id not in document_ids_with_exclamation:
                        document_ids_with_exclamation.append(document_id)

            if len(selected_sections) >= max_sections:
                break

        if not selected_sections:
            logger.warning(
                "No valid sections selected from LLM response, returning first max_sections"
            )
            return sections[:max_sections], None

        selected_document_ids = [
            section.center_chunk.document_id for section in selected_sections
        ]
        logger.debug(
            "LLM selected %s valid sections from %s total candidates. "
            "Selected document IDs: %s. Document IDs with exclamation: %s",
            len(selected_sections),
            len(sections),
            selected_document_ids,
            document_ids_with_exclamation if document_ids_with_exclamation else [],
        )

        return selected_sections, (
            document_ids_with_exclamation if document_ids_with_exclamation else None
        )

    except Exception as e:
        logger.error("Error calling LLM for document selection: %s", e)
        return sections[:max_sections], None
