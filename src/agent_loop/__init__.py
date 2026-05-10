"""Agent loop utilities for local generation workflows.

The lightweight runtime loops are imported eagerly so their registry names are
available immediately. Training/data helpers are resolved lazily to avoid
pulling in heavier optional dependencies, such as torch, for simple CLI smoke
tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agent_loop import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    RolloutStep,
    get_registered_agent_loop,
    list_registered_agent_loops,
    register,
    simple_timer,
)
from .context import AgentContext, SearchContext, SearchResult
from .evaluation import (
    QueryEvaluation,
    SearchEvaluationConfig,
    SearchResultEvaluator,
    SearchRoundEvaluation,
)
from .plain_generation_loop import PlainGenerationLoop, PlainGenerationLoopConfig
from .search_agent_loop import (
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchRoundResult,
    SearchToolCall,
    build_search_agent_instruction,
)
from .search_client import SearchClient, SearchClientConfig
from .sft import SFTExample, build_search_sft_example
from .single_turn_agent_loop import SingleTurnAgentLoop, SingleTurnAgentLoopConfig
from .tool import FunctionTool, Tool, ToolSchema
from .tool_agent_loop import ToolAgentLoop, ToolAgentLoopConfig
from .tool_parser import (
    FunctionCall,
    HermesToolParser,
    JSONToolParser,
    Llama3ToolParser,
    ToolParser,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # data.py imports torch; keep it out of lightweight runtime imports.
    "DEFAULT_TOOL_SYSTEM_PROMPT": (".data", "DEFAULT_TOOL_SYSTEM_PROMPT"),
    "PromptBatch": (".data", "PromptBatch"),
    "PromptOnlyDataset": (".data", "PromptOnlyDataset"),
    "PromptSample": (".data", "PromptSample"),
    "PromptTrainingExample": (".data", "PromptTrainingExample"),
    "build_prompt_dataloader": (".data", "build_prompt_dataloader"),
    "build_prompt_ids_from_messages": (".data", "build_prompt_ids_from_messages"),
    "build_prompt_messages": (".data", "build_prompt_messages"),
    "collate_prompt_batch": (".data", "collate_prompt_batch"),
    "normalize_prompt_training_example": (".data", "normalize_prompt_training_example"),
    "prompt_batch_to_search_batch": (".data", "prompt_batch_to_search_batch"),
    # grpo.py depends on data.py and reward.py.
    "GRPOAdvantageConfig": (".grpo", "GRPOAdvantageConfig"),
    "GRPORolloutSample": (".grpo", "GRPORolloutSample"),
    "PromptGroupSamplingConfig": (".grpo", "PromptGroupSamplingConfig"),
    "ScoredGRPORollout": (".grpo", "ScoredGRPORollout"),
    "build_grpo_sampling_params": (".grpo", "build_grpo_sampling_params"),
    "compute_grpo_outcome_advantage": (".grpo", "compute_grpo_outcome_advantage"),
    "sample_prompt_group": (".grpo", "sample_prompt_group"),
    "score_prompt_group": (".grpo", "score_prompt_group"),
    # Reward is light today, but lazy keeps the public surface grouped.
    "BatchJudgeFn": (".reward", "BatchJudgeFn"),
    "JudgeFn": (".reward", "JudgeFn"),
    "SearchRewardConfig": (".reward", "SearchRewardConfig"),
    "SearchRewardFunction": (".reward", "SearchRewardFunction"),
    "normalize_answer_text": (".reward", "normalize_answer_text"),
    "simple_sparse_correctness_reward": (".reward", "simple_sparse_correctness_reward"),
    # Intent model imports torch only when trained/loaded, but it is not needed
    # for plain local generation.
    "INTENT_LABELS": (".intent_classifier", "INTENT_LABELS"),
    "IntentPipeline": (".intent_classifier", "IntentPipeline"),
    "IntentPrediction": (".intent_classifier", "IntentPrediction"),
    "IntentionClassificationPipeline": (
        ".intent_classifier",
        "IntentionClassificationPipeline",
    ),
    "load_intent_training_data": (".intent_classifier", "load_training_data"),
    "resolve_search_settings": (".intent_classifier", "resolve_search_settings"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AgentContext",
    "GRPOAdvantageConfig",
    "build_grpo_sampling_params",
    "compute_grpo_outcome_advantage",
    "build_prompt_dataloader",
    "build_prompt_ids_from_messages",
    "build_prompt_messages",
    "prompt_batch_to_search_batch",
    "BatchJudgeFn",
    "JudgeFn",
    "SearchRewardConfig",
    "SearchRewardFunction",
    "normalize_answer_text",
    "simple_sparse_correctness_reward",
    "AgentLoopBase",
    "AgentLoopConfig",
    "AgentLoopOutput",
    "PlainGenerationLoop",
    "PlainGenerationLoopConfig",
    "RolloutStep",
    "build_search_agent_instruction",
    "collate_prompt_batch",
    "DEFAULT_TOOL_SYSTEM_PROMPT",
    "FunctionCall",
    "FunctionTool",
    "HermesToolParser",
    "INTENT_LABELS",
    "IntentPipeline",
    "IntentPrediction",
    "IntentionClassificationPipeline",
    "load_intent_training_data",
    "GRPORolloutSample",
    "PromptGroupSamplingConfig",
    "resolve_search_settings",
    "JSONToolParser",
    "Llama3ToolParser",
    "normalize_prompt_training_example",
    "PromptBatch",
    "PromptOnlyDataset",
    "PromptSample",
    "PromptTrainingExample",
    "QueryEvaluation",
    "SearchAgentLoop",
    "SearchAgentLoopConfig",
    "SearchRoundResult",
    "SearchToolCall",
    "SFTExample",
    "SearchClient",
    "SearchClientConfig",
    "SearchContext",
    "SearchEvaluationConfig",
    "SearchResult",
    "SearchResultEvaluator",
    "SearchRoundEvaluation",
    "ScoredGRPORollout",
    "SingleTurnAgentLoop",
    "SingleTurnAgentLoopConfig",
    "build_search_sft_example",
    "sample_prompt_group",
    "score_prompt_group",
    "Tool",
    "ToolAgentLoop",
    "ToolAgentLoopConfig",
    "ToolParser",
    "ToolSchema",
    "get_registered_agent_loop",
    "list_registered_agent_loops",
    "register",
    "simple_timer",
]
