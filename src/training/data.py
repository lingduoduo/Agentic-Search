"""Prompt-only training data helpers for tool-use / agentic RL workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_TOOL_SYSTEM_PROMPT = (
    "You are a tool-using assistant. "
    "Use tools when they help, and answer directly when the question is already solved."
)


def normalize_question_text(question: Any) -> str:
    """Normalize raw QA questions into stable, searchable prompts."""

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    if not normalized:
        raise ValueError("Training example is missing a non-empty `question`.")
    if normalized[-1] != "?":
        normalized = normalized.rstrip(".!;:") + "?"
    return normalized


def normalize_answer_aliases(answers: Any) -> list[str]:
    """Return de-duplicated answer aliases, preserving source order."""

    if answers is None:
        return []
    if isinstance(answers, str):
        candidates = [answers]
    elif isinstance(answers, Sequence):
        candidates = [str(answer) for answer in answers]
    else:
        candidates = [str(answers)]

    aliases: list[str] = []
    seen: set[str] = set()
    for answer in candidates:
        alias = re.sub(r"\s+", " ", answer).strip()
        if not alias:
            continue
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def extract_answer_aliases(example: Mapping[str, Any]) -> list[str]:
    """Extract answer aliases from common QA dataset field names."""

    for key in ("golden_answers", "answers", "answer_aliases", "ground_truth"):
        aliases = normalize_answer_aliases(example.get(key))
        if aliases:
            return aliases
    reward_model = example.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, Mapping):
            aliases = normalize_answer_aliases(ground_truth.get("target"))
            if aliases:
                return aliases
    answer = str(example.get("answer", "")).strip()
    return normalize_answer_aliases(answer)


def extract_question_text(example: Mapping[str, Any]) -> str:
    """Extract a question from either a direct field or a prompt message."""

    question = str(example.get("question", "")).strip()
    if question:
        return question
    prompt = example.get("prompt")
    if prompt is not None:
        messages = normalize_prompt_messages(prompt)
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return message["content"]
        if messages:
            return messages[-1]["content"]
    return ""


def build_search_qa_prompt(question: str, *, template_type: str = "base") -> str:
    """Normalize a QA question for search-agent datasets."""

    if template_type != "base":
        raise NotImplementedError(f"Unsupported QA prompt template: {template_type}")
    return normalize_question_text(question)


def build_search_qa_messages(
    question: str,
    *,
    template_type: str = "base",
    system_prompt: str | None = None,
    max_search_limit: int = 5,
    max_url_fetch: int = 3,
) -> list[dict[str, str]]:
    """Build QA messages using the same prompt contract as SearchAgentLoop."""

    if template_type != "base":
        raise NotImplementedError(f"Unsupported QA prompt template: {template_type}")
    if system_prompt is None:
        from src.agents.search import build_search_agent_instruction

        system_prompt = build_search_agent_instruction(
            max_search_limit=max_search_limit,
            max_url_fetch=max_url_fetch,
        )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": normalize_question_text(question)},
    ]


def build_search_qa_record(
    example: Mapping[str, Any],
    *,
    split: str | None = None,
    index: int | None = None,
    data_source: str = "qa",
    ability: str = "fact-reasoning",
    template_type: str = "base",
) -> dict[str, Any]:
    """Convert a raw QA row into a rollout/reward-ready training record.

    The output mirrors the lightweight structure used by VERL-style pipelines
    while keeping fields directly consumable by this repo's PromptOnlyDataset.
    """

    question = normalize_question_text(example.get("question", ""))
    answer_aliases = extract_answer_aliases(example)
    if not answer_aliases:
        raise ValueError("Training example is missing non-empty answer aliases.")

    extra_info: dict[str, Any] = {}
    if split is not None:
        extra_info["split"] = split
    if index is not None:
        extra_info["index"] = index

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": build_search_qa_prompt(
                    question,
                    template_type=template_type,
                ),
            }
        ],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": answer_aliases},
        },
        "extra_info": extra_info,
    }


def format_rag_reference(retrieval_result: Sequence[Mapping[str, Any]]) -> str:
    """Format retrieved documents as numbered context references."""

    references: list[str] = []
    for index, doc_item in enumerate(retrieval_result, start=1):
        content = str(doc_item.get("contents", "")).strip()
        title = str(doc_item.get("title", "")).strip()
        text = content
        if content:
            lines = content.splitlines()
            if not title and lines:
                title = lines[0].strip()
                text = "\n".join(lines[1:]).strip()
        if not title:
            title = str(doc_item.get("id", f"Doc {index}")).strip()
        references.append(f"Doc {index}(Title: {title}) {text}".strip())
    return "\n".join(references)


def build_search_rag_prompt(
    question: str,
    context: str,
    *,
    template_type: str = "base",
) -> str:
    """Build a compact RAG prompt with fixed question/context sections."""

    if template_type != "base":
        raise NotImplementedError(f"Unsupported RAG prompt template: {template_type}")
    return (
        "Answer the given question with the potentially useful context.\n"
        "Reason in <think></think> and return the final answer in "
        "<answer></answer>.\n"
        f"Question: {normalize_question_text(question)}\n"
        f"Context:\n{context.strip()}\n"
    )


def build_search_rag_record(
    example: Mapping[str, Any],
    *,
    context: str | None = None,
    split: str | None = None,
    index: int | None = None,
    data_source: str = "qa",
    ability: str = "fact-reasoning",
    template_type: str = "base",
) -> dict[str, Any]:
    """Convert a QA row plus retrieval context into a compact RAG record."""

    question = normalize_question_text(example.get("question", ""))
    answer_aliases = extract_answer_aliases(example)
    if not answer_aliases:
        raise ValueError("Training example is missing non-empty answer aliases.")
    resolved_context = str(
        context if context is not None else example.get("context", "")
    )
    if not resolved_context.strip():
        raise ValueError("RAG training example is missing non-empty `context`.")

    extra_info: dict[str, Any] = {}
    if split is not None:
        extra_info["split"] = split
    if index is not None:
        extra_info["index"] = index

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": build_search_rag_prompt(
                    question,
                    resolved_context,
                    template_type=template_type,
                ),
            }
        ],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": answer_aliases},
        },
        "extra_info": extra_info,
    }


def make_search_qa_map_fn(
    split: str,
    *,
    data_source: str = "qa",
    ability: str = "fact-reasoning",
    template_type: str = "base",
) -> Callable[[Mapping[str, Any], int], dict[str, Any]]:
    """Return a datasets.map function for QA-to-search-agent records."""

    def process_fn(example: Mapping[str, Any], index: int) -> dict[str, Any]:
        return build_search_qa_record(
            example,
            split=split,
            index=index,
            data_source=data_source,
            ability=ability,
            template_type=template_type,
        )

    return process_fn


def make_search_rag_map_fn(
    split: str,
    *,
    retrieval_cache: Mapping[str, Sequence[Mapping[str, Any]]],
    corpus: Mapping[str, Mapping[str, Any]],
    topk: int = 3,
    data_source: str = "qa",
    ability: str = "fact-reasoning",
    template_type: str = "base",
) -> Callable[[Mapping[str, Any], int], dict[str, Any]]:
    """Return a datasets.map function for QA plus cached retrieval context."""

    if topk < 1:
        raise ValueError("topk must be at least 1.")

    def process_fn(example: Mapping[str, Any], index: int) -> dict[str, Any]:
        question = str(example.get("question", "")).strip()
        if question not in retrieval_cache:
            raise KeyError(f"Missing retrieval cache entry for question: {question}")
        retrieval_docs: list[Mapping[str, Any]] = []
        for hit in retrieval_cache[question][:topk]:
            doc_id = str(hit.get("id", ""))
            if doc_id not in corpus:
                raise KeyError(f"Missing corpus document for retrieval id: {doc_id}")
            retrieval_docs.append(corpus[doc_id])

        return build_search_rag_record(
            example,
            context=format_rag_reference(retrieval_docs),
            split=split,
            index=index,
            data_source=data_source,
            ability=ability,
            template_type=template_type,
        )

    return process_fn


@dataclass(frozen=True)
class PromptTrainingExample:
    """Raw training example for prompt-only rollout generation."""

    question: str
    ground_truth: str
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSample:
    """One tokenized prompt example ready for rollout generation."""

    question: str
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    ground_truth: str
    tools: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBatch:
    """A padded batch of prompt-only inputs for RL/SFT rollout generation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: list[int]
    questions: list[str]
    messages: list[list[dict[str, Any]]]
    ground_truths: list[str]
    tools: list[list[str]]
    metadata: list[dict[str, Any]]


def normalize_prompt_training_example(
    example: PromptTrainingExample | Mapping[str, Any],
) -> PromptTrainingExample:
    """Convert a dict-like record into a validated PromptTrainingExample."""

    if isinstance(example, PromptTrainingExample):
        return example

    qa_style_record = any(
        key in example
        for key in ("golden_answers", "answers", "answer_aliases", "answer", "prompt")
    )
    if qa_style_record:
        question = normalize_question_text(extract_question_text(example))
    else:
        question = re.sub(r"\s+", " ", str(example.get("question", ""))).strip()
        if not question:
            raise ValueError("Training example is missing a non-empty `question`.")
    answer_aliases = extract_answer_aliases(example)
    ground_truth = answer_aliases[0] if answer_aliases else ""
    tools = [str(tool) for tool in example.get("tools", [])]
    metadata = {
        key: value
        for key, value in dict(example).items()
        if key
        not in {
            "question",
            "ground_truth",
            "golden_answers",
            "answers",
            "answer_aliases",
            "answer",
            "tools",
        }
    }
    if len(answer_aliases) > 1:
        metadata.setdefault("answer_aliases", answer_aliases)

    if not ground_truth:
        raise ValueError("Training example is missing a non-empty `ground_truth`.")

    return PromptTrainingExample(
        question=question,
        ground_truth=ground_truth,
        tools=tools,
        metadata=metadata,
    )


def build_prompt_messages(
    question: str,
    *,
    tools: Sequence[str] | None = None,
    system_prompt: str | None = DEFAULT_TOOL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build a rollout prompt from question + available tool names.

    The dataset intentionally contains only the prompt-side context, not the
    intermediate reasoning or action trace.
    """

    messages: list[dict[str, str]] = []
    resolved_tools = [tool for tool in (tools or []) if tool]
    if system_prompt:
        content = system_prompt.strip()
        if resolved_tools:
            content += "\n\nAvailable tools: " + ", ".join(resolved_tools)
        messages.append({"role": "system", "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def normalize_prompt_messages(prompt: Any) -> list[dict[str, str]]:
    """Normalize stored prompt fields into chat messages."""

    if isinstance(prompt, str):
        content = prompt.strip()
        if not content:
            raise ValueError("Stored prompt string is empty.")
        return [{"role": "user", "content": content}]
    if isinstance(prompt, Sequence):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, Mapping):
                raise ValueError("Stored prompt messages must be mappings.")
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        if messages:
            return messages
    raise ValueError("Stored prompt must be a string or non-empty message list.")


def build_prompt_ids_from_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    prompt_length: int = 4096,
) -> list[int]:
    """Tokenize a chat prompt using the same fallback logic as AgentLoopBase."""

    chat_template = getattr(tokenizer, "chat_template", "__missing__")
    if hasattr(tokenizer, "apply_chat_template") and chat_template is not None:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return list(prompt_ids)[-prompt_length:]

    joined = "\n".join(message.get("content", "") for message in messages)
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(joined))[-prompt_length:]
    raise TypeError("tokenizer must implement apply_chat_template(...) or encode(...).")


class PromptOnlyDataset(Dataset[PromptSample]):
    """Dataset that maps raw examples to tokenized prompt-only rollout inputs."""

    def __init__(
        self,
        examples: Sequence[PromptTrainingExample | Mapping[str, Any]],
        *,
        tokenizer: Any,
        prompt_length: int = 4096,
        prompt_builder: Callable[[PromptTrainingExample], list[dict[str, Any]]]
        | None = None,
    ) -> None:
        self.examples = [
            normalize_prompt_training_example(example) for example in examples
        ]
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.prompt_builder = prompt_builder

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PromptSample:
        example = self.examples[index]
        if self.prompt_builder is not None:
            messages = list(self.prompt_builder(example))
        elif "prompt" in example.metadata:
            messages = normalize_prompt_messages(example.metadata["prompt"])
        else:
            messages = build_prompt_messages(
                example.question,
                tools=example.tools,
            )
        prompt_ids = build_prompt_ids_from_messages(
            self.tokenizer,
            messages,
            prompt_length=self.prompt_length,
        )
        return PromptSample(
            question=example.question,
            messages=messages,
            prompt_ids=prompt_ids,
            ground_truth=example.ground_truth,
            tools=list(example.tools),
            metadata=dict(example.metadata),
        )


def collate_prompt_batch(
    samples: Sequence[PromptSample],
    *,
    pad_token_id: int = 0,
) -> PromptBatch:
    """Pad tokenized prompt samples into a rollout batch."""

    if not samples:
        raise ValueError("Cannot collate an empty prompt batch.")

    max_length = max(len(sample.prompt_ids) for sample in samples)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    prompt_lengths: list[int] = []

    for sample in samples:
        length = len(sample.prompt_ids)
        prompt_lengths.append(length)
        padding = max_length - length
        # Left-pad so every prompt ends at position [-1], aligning generation
        # starts without requiring per-sample index arithmetic.
        input_ids.append([pad_token_id] * padding + list(sample.prompt_ids))
        attention_masks.append([0] * padding + [1] * length)

    return PromptBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_masks, dtype=torch.long),
        prompt_lengths=prompt_lengths,
        questions=[sample.question for sample in samples],
        messages=[list(sample.messages) for sample in samples],
        ground_truths=[sample.ground_truth for sample in samples],
        tools=[list(sample.tools) for sample in samples],
        metadata=[dict(sample.metadata) for sample in samples],
    )


def prompt_batch_to_search_batch(
    batch: PromptBatch,
) -> Any:
    """Convert a PromptBatch into a rollout-ready SearchBatch.

    The result matches the `gen_batch` shape expected by
    `LLMGenerationManager.run_llm_loop(...)`: prompt token tensors live in
    `.batch`, while prompt-only supervision fields stay in `.non_tensor_batch`.
    Intermediate reasoning or tool traces are intentionally absent.
    """

    from src import SearchBatch

    search_batch = SearchBatch.from_dict(
        {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "position_ids": torch.arange(
                batch.input_ids.shape[1],
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(batch.input_ids),
        }
    )
    search_batch.non_tensor_batch = {
        "question": list(batch.questions),
        "golden_answers": [[ground_truth] for ground_truth in batch.ground_truths],
        "ground_truth": list(batch.ground_truths),
        "messages": [list(messages) for messages in batch.messages],
        "tools": [list(tools) for tools in batch.tools],
        "metadata": [dict(item) for item in batch.metadata],
    }
    search_batch.meta_info["prompt_lengths"] = list(batch.prompt_lengths)
    return search_batch


def build_prompt_dataloader(
    examples: Sequence[PromptTrainingExample | Mapping[str, Any]],
    *,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool = False,
    prompt_length: int = 4096,
    pad_token_id: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
    prompt_builder: Callable[[PromptTrainingExample], list[dict[str, Any]]]
    | None = None,
) -> DataLoader:
    """Create a DataLoader that yields left-padded prompt-only rollout batches."""

    dataset = PromptOnlyDataset(
        examples,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        prompt_builder=prompt_builder,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda batch: collate_prompt_batch(batch, pad_token_id=pad_token_id),
    )
