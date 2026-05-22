"""Top-level package for Agentic-Search.

Lightweight runtime loops are imported eagerly so their registry names are
available immediately. Training/data helpers are loaded lazily to avoid pulling
in torch for simple CLI smoke tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agents.base import AgentLoopBase as AgentLoopBase
from .agents.base import AgentLoopConfig as AgentLoopConfig
from .agents.base import AgentLoopOutput as AgentLoopOutput
from .agents.base import RolloutStep as RolloutStep
from .agents.base import get_registered_agent_loop as get_registered_agent_loop
from .agents.base import list_registered_agent_loops as list_registered_agent_loops
from .agents.base import register as register
from .agents.base import simple_timer as simple_timer
from .agents.plain import PlainGenerationLoop as PlainGenerationLoop
from .agents.plain import PlainGenerationLoopConfig as PlainGenerationLoopConfig
from .agents.search import SearchAgentLoop as SearchAgentLoop
from .agents.search import SearchAgentLoopConfig as SearchAgentLoopConfig
from .agents.search import SearchRoundResult as SearchRoundResult
from .agents.search import SearchToolCall as SearchToolCall
from .agents.search import (
    build_search_agent_instruction as build_search_agent_instruction,
)
from .agents.single_turn import SingleTurnAgentLoop as SingleTurnAgentLoop
from .agents.single_turn import SingleTurnAgentLoopConfig as SingleTurnAgentLoopConfig
from .agents.state import AgentState as AgentState
from .agents.state import PerformanceMetrics as PerformanceMetrics
from .agents.state import Plan as Plan
from .agents.state import PlanStep as PlanStep
from .agents.state import RetrievedDocument as RetrievedDocument
from .agents.state import RouteDecision as RouteDecision
from .agents.state import TaskNode as TaskNode
from .agents.state import TaskStatus as TaskStatus
from .agents.state import TaskType as TaskType
from .agents.state import ToolCall as ToolCall
from .agents.state import ToolExecutionResult as ToolExecutionResult
from .agents.state import ToolResult as ToolResult
from .agents.state import ToolType as ToolType
from .agents.state import UserRequest as UserRequest
from .agents.tool_calling import ToolAgentLoop as ToolAgentLoop
from .agents.tool_calling import ToolAgentLoopConfig as ToolAgentLoopConfig
from .connectors import BaseConnector as BaseConnector
from .connectors import CheckpointedConnector as CheckpointedConnector
from .connectors import (
    CheckpointedConnectorWithPermSync as CheckpointedConnectorWithPermSync,
)
from .connectors import ConnectorCheckpoint as ConnectorCheckpoint
from .connectors import ConnectorFailure as ConnectorFailure
from .connectors import Document as Document
from .connectors import InMemoryConnector as InMemoryConnector
from .connectors import LoadConnector as LoadConnector
from .connectors import LocalFileConnector as LocalFileConnector
from .connectors import OAuthConnector as OAuthConnector
from .connectors import PollConnector as PollConnector
from .connectors import SearchConnector as SearchConnector
from .connectors import SlimConnector as SlimConnector
from .connectors import SlimConnectorWithPermSync as SlimConnectorWithPermSync
from .connectors import SlimDocument as SlimDocument
from .connectors import StaticCredentialsProvider as StaticCredentialsProvider
from .retrieval.client import SearchClient as SearchClient
from .retrieval.client import SearchClientConfig as SearchClientConfig
from .retrieval.context import AgentContext as AgentContext
from .retrieval.context import SearchContext as SearchContext
from .retrieval.context import SearchResult as SearchResult
from .retrieval.sparse_retriever import SparseRetriever as SparseRetriever
from .retrieval.sparse_retriever import SparseRetrieverConfig as SparseRetrieverConfig
from .tools.base import FunctionTool as FunctionTool
from .tools.base import Tool as Tool
from .tools.base import ToolSchema as ToolSchema
from .tools.parsers import FunctionCall as FunctionCall
from .tools.parsers import HermesToolParser as HermesToolParser
from .tools.parsers import JSONToolParser as JSONToolParser
from .tools.parsers import Llama3ToolParser as Llama3ToolParser
from .tools.parsers import ToolParser as ToolParser
from .training.evaluation import QueryEvaluation as QueryEvaluation
from .training.evaluation import SearchEvaluationConfig as SearchEvaluationConfig
from .training.evaluation import SearchResultEvaluator as SearchResultEvaluator
from .training.evaluation import SearchRoundEvaluation as SearchRoundEvaluation
from .training.sft import SFTExample as SFTExample
from .training.sft import build_search_sft_example as build_search_sft_example

# Torch-heavy modules — loaded on first access, never at import time.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # training.data
    "DEFAULT_TOOL_SYSTEM_PROMPT": (".training.data", "DEFAULT_TOOL_SYSTEM_PROMPT"),
    "PromptBatch": (".training.data", "PromptBatch"),
    "PromptOnlyDataset": (".training.data", "PromptOnlyDataset"),
    "PromptSample": (".training.data", "PromptSample"),
    "PromptTrainingExample": (".training.data", "PromptTrainingExample"),
    "build_prompt_dataloader": (".training.data", "build_prompt_dataloader"),
    "build_prompt_ids_from_messages": (
        ".training.data",
        "build_prompt_ids_from_messages",
    ),
    "build_prompt_messages": (".training.data", "build_prompt_messages"),
    "build_search_rag_record": (".training.data", "build_search_rag_record"),
    "build_search_qa_messages": (".training.data", "build_search_qa_messages"),
    "build_search_qa_prompt": (".training.data", "build_search_qa_prompt"),
    "build_search_qa_record": (".training.data", "build_search_qa_record"),
    "collate_prompt_batch": (".training.data", "collate_prompt_batch"),
    "format_rag_reference": (".training.data", "format_rag_reference"),
    "make_search_rag_map_fn": (".training.data", "make_search_rag_map_fn"),
    "make_search_qa_map_fn": (".training.data", "make_search_qa_map_fn"),
    "normalize_prompt_training_example": (
        ".training.data",
        "normalize_prompt_training_example",
    ),
    "normalize_question_text": (".training.data", "normalize_question_text"),
    "prompt_batch_to_search_batch": (".training.data", "prompt_batch_to_search_batch"),
    # training.grpo
    "GRPOAdvantageConfig": (".training.grpo", "GRPOAdvantageConfig"),
    "GRPORolloutSample": (".training.grpo", "GRPORolloutSample"),
    "PromptGroupSamplingConfig": (".training.grpo", "PromptGroupSamplingConfig"),
    "ScoredGRPORollout": (".training.grpo", "ScoredGRPORollout"),
    "build_grpo_sampling_params": (".training.grpo", "build_grpo_sampling_params"),
    "compute_grpo_outcome_advantage": (
        ".training.grpo",
        "compute_grpo_outcome_advantage",
    ),
    "sample_prompt_group": (".training.grpo", "sample_prompt_group"),
    "score_prompt_group": (".training.grpo", "score_prompt_group"),
    # training.reward
    "BatchJudgeFn": (".training.reward", "BatchJudgeFn"),
    "JudgeFn": (".training.reward", "JudgeFn"),
    "SearchRewardConfig": (".training.reward", "SearchRewardConfig"),
    "SearchRewardFunction": (".training.reward", "SearchRewardFunction"),
    "normalize_answer_text": (".training.reward", "normalize_answer_text"),
    "simple_sparse_correctness_reward": (
        ".training.reward",
        "simple_sparse_correctness_reward",
    ),
    # model.intent_classifier
    "INTENT_LABELS": (".model.intent_classifier", "INTENT_LABELS"),
    "IntentPipeline": (".model.intent_classifier", "IntentPipeline"),
    "IntentPrediction": (".model.intent_classifier", "IntentPrediction"),
    "IntentionClassificationPipeline": (
        ".model.intent_classifier",
        "IntentionClassificationPipeline",
    ),
    "load_intent_training_data": (".model.intent_classifier", "load_training_data"),
    "resolve_search_settings": (".model.intent_classifier", "resolve_search_settings"),
    # model.intent_training
    "IntentTrainingResult": (".model.intent_training", "IntentTrainingResult"),
    "build_examples_for_document": (
        ".model.intent_training",
        "build_examples_for_document",
    ),
    "generate_intent_examples": (".model.intent_training", "generate_intent_examples"),
    "train_intent_classifier": (".model.intent_training", "train_intent_classifier"),
    "write_intent_examples": (".model.intent_training", "write_intent_examples"),
    # model.generation (SearchToolCall excluded — already exported from agents.search)
    "ActorRolloutStep": (".model.generation", "ActorRolloutStep"),
    "ContinuationDecision": (".model.generation", "ContinuationDecision"),
    "FinalGenBatchOutput": (".model.generation", "FinalGenBatchOutput"),
    "GenerationConfig": (".model.generation", "GenerationConfig"),
    "GRPOPromptGroupResult": (".model.generation", "GRPOPromptGroupResult"),
    "GRPORolloutSafetyConfig": (".model.generation", "GRPORolloutSafetyConfig"),
    "GRPOTrainingStepResult": (".model.generation", "GRPOTrainingStepResult"),
    "GroupedRolloutBatch": (".model.generation", "GroupedRolloutBatch"),
    "LLMGenerationManager": (".model.generation", "LLMGenerationManager"),
    "PPOPolicyLossConfig": (".model.generation", "PPOPolicyLossConfig"),
    "ReActContextTransition": (".model.generation", "ReActContextTransition"),
    "ScoredGroupedRollout": (".model.generation", "ScoredGroupedRollout"),
    "SearchBatch": (".model.generation", "SearchBatch"),
    "SearchDecision": (".model.generation", "SearchDecision"),
    "SearchStep": (".model.generation", "SearchStep"),
    "SearchTrajectoryLog": (".model.generation", "SearchTrajectoryLog"),
    "apply_rollout_safety_penalties": (
        ".model.generation",
        "apply_rollout_safety_penalties",
    ),
    "apply_safety_penalties_to_scored_rollouts": (
        ".model.generation",
        "apply_safety_penalties_to_scored_rollouts",
    ),
    "assign_group_relative_advantages": (
        ".model.generation",
        "assign_group_relative_advantages",
    ),
    "async_run_grpo_training_step": (
        ".model.generation",
        "async_run_grpo_training_step",
    ),
    "async_run_prompt_rollout_group": (
        ".model.generation",
        "async_run_prompt_rollout_group",
    ),
    "compute_trajectory_policy_loss": (
        ".model.generation",
        "compute_trajectory_policy_loss",
    ),
    "compute_reinforce_policy_loss": (
        ".training.ppo.core_algos",
        "compute_reinforce_policy_loss",
    ),
    "compute_reinforce_policy_loss_core": (
        ".training.ppo.core_algos",
        "compute_reinforce_policy_loss_core",
    ),
    "format_group_rollout": (".model.generation", "format_group_rollout"),
    "format_scored_group_rollout": (".model.generation", "format_scored_group_rollout"),
    "format_search_trajectory_log": (
        ".model.generation",
        "format_search_trajectory_log",
    ),
    "format_trajectory_batch": (".model.generation", "format_trajectory_batch"),
    "save_training_batch_jsonl": (".model.generation", "save_training_batch_jsonl"),
    "score_group_rollout": (".model.generation", "score_group_rollout"),
    "trajectory_log_prob_pack": (".model.generation", "trajectory_log_prob_pack"),
    # training.ppo.controller
    "LocalGRPOController": (".training.ppo.controller", "LocalGRPOController"),
    "RolloutResult": (".training.ppo.controller", "RolloutResult"),
    # training.ppo.reward_manager
    "PPORewardManager": (".training.ppo.reward_manager", "PPORewardManager"),
    "qa_exact_match_score": (
        ".training.ppo.reward_manager",
        "qa_exact_match_score",
    ),
    "select_reward_score_fn": (
        ".training.ppo.reward_manager",
        "select_reward_score_fn",
    ),
    # model.tensor_helper
    "TensorConfig": (".model.tensor_helper", "TensorConfig"),
    "TensorHelper": (".model.tensor_helper", "TensorHelper"),
}

__all__ = sorted(
    [
        n
        for n in globals()
        if not n.startswith("_") and n not in {"import_module", "Any"}
    ]
    + list(_LAZY_EXPORTS)
)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
