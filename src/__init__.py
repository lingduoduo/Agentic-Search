"""Top-level package for Agentic-Search.

Lightweight runtime loops are imported eagerly so their registry names are
available immediately. Training/data helpers are loaded lazily to avoid pulling
in torch for simple CLI smoke tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agents.core.base import AgentLoopBase as AgentLoopBase
from .agents.core.base import AgentLoopConfig as AgentLoopConfig
from .agents.core.base import AgentLoopOutput as AgentLoopOutput
from .agents.core.base import RolloutStep as RolloutStep
from .agents.core.base import CANONICAL_AGENT_NAMES as CANONICAL_AGENT_NAMES
from .agents.core.base import get_registered_agent_loop as get_registered_agent_loop
from .agents.core.base import list_registered_agent_loops as list_registered_agent_loops
from .agents.core.base import register as register
from .agents.core.base import resolve_agent_name as resolve_agent_name
from .agents.core.base import simple_timer as simple_timer
from .agents.generation import PlainGenerationLoop as PlainGenerationLoop
from .agents.generation import PlainGenerationLoopConfig as PlainGenerationLoopConfig
from .agents.search import SearchAgentLoop as SearchAgentLoop
from .agents.search import SearchAgentLoopConfig as SearchAgentLoopConfig
from .agents.search import SearchRoundResult as SearchRoundResult
from .agents.search import SearchToolCall as SearchToolCall
from .agents.search import (
    build_search_agent_instruction as build_search_agent_instruction,
)
from .agents.generation import SingleTurnAgentLoop as SingleTurnAgentLoop
from .agents.generation import SingleTurnAgentLoopConfig as SingleTurnAgentLoopConfig
from .agents.core.state import AgentState as AgentState
from .agents.core.state import PerformanceMetrics as PerformanceMetrics
from .agents.core.state import Plan as Plan
from .agents.core.state import PlanStep as PlanStep
from .agents.core.state import RetrievedDocument as RetrievedDocument
from .agents.core.state import RouteDecision as RouteDecision
from .agents.core.state import TaskNode as TaskNode
from .agents.core.state import TaskStatus as TaskStatus
from .agents.core.state import TaskType as TaskType
from .agents.core.state import ToolCall as ToolCall
from .agents.core.state import ToolExecutionResult as ToolExecutionResult
from .agents.core.state import ToolResult as ToolResult
from .agents.core.state import ToolType as ToolType
from .agents.core.state import UserRequest as UserRequest
from .agents.tool import ToolAgentLoop as ToolAgentLoop
from .agents.tool import ToolAgentLoopConfig as ToolAgentLoopConfig
from .internal.connectors import ConnectorCheckpoint as ConnectorCheckpoint
from .internal.connectors import ConnectorFailure as ConnectorFailure
from .internal.connectors import Document as Document
from .internal.connectors import HierarchyNode as HierarchyNode
from .internal.connectors import SlimDocument as SlimDocument
from .context.retrieval.client import SearchClient as SearchClient
from .context.retrieval.client import SearchClientConfig as SearchClientConfig
from .context.search import AgentContext as AgentContext
from .context.search import SearchContext as SearchContext
from .context.search import SearchResult as SearchResult
from .internal.document_index.retrieval import SparseRetriever as SparseRetriever
from .internal.document_index.retrieval import (
    SparseRetrieverConfig as SparseRetrieverConfig,
)
from .tools.base import FunctionTool as FunctionTool
from .tools.base import Tool as Tool
from .tools.base import ToolSchema as ToolSchema
from .tools.parsers import FunctionCall as FunctionCall
from .tools.parsers import HermesToolParser as HermesToolParser
from .tools.parsers import JSONToolParser as JSONToolParser
from .tools.parsers import Llama3ToolParser as Llama3ToolParser
from .tools.parsers import ToolParser as ToolParser

# Torch-heavy modules — loaded on first access, never at import time.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # training.evaluation (pulls torch transitively — keep lazy)
    "QueryEvaluation": (".training.evaluation", "QueryEvaluation"),
    "SearchEvaluationConfig": (".training.evaluation", "SearchEvaluationConfig"),
    "SearchResultEvaluator": (".training.evaluation", "SearchResultEvaluator"),
    "SearchRoundEvaluation": (".training.evaluation", "SearchRoundEvaluation"),
    # training.sft (imports torch at module level)
    "SFTExample": (".training.sft", "SFTExample"),
    "build_search_sft_example": (".training.sft", "build_search_sft_example"),
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
    "sample_prompt_batch": (".training.grpo", "sample_prompt_batch"),
    "score_prompt_group": (".training.grpo", "score_prompt_group"),
    "score_prompt_batch": (".training.grpo", "score_prompt_batch"),
    "OnPolicyGRPOConfig": (".training.grpo", "OnPolicyGRPOConfig"),
    "OnPolicyBatchStats": (".training.grpo", "OnPolicyBatchStats"),
    "filter_zero_advantage_groups": (".training.grpo", "filter_zero_advantage_groups"),
    "assemble_on_policy_batch": (".training.grpo", "assemble_on_policy_batch"),
    "compute_on_policy_batch_stats": (
        ".training.grpo",
        "compute_on_policy_batch_stats",
    ),
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
    # servers
    "OnlineSearchConfig": (".internal.servers.web_search.google", "OnlineSearchConfig"),
    "OnlineSearchEngine": (".internal.servers.web_search.google", "OnlineSearchEngine"),
    "SerpSearchConfig": (".internal.servers.web_search.serp", "SerpSearchConfig"),
    "SerpSearchEngine": (".internal.servers.web_search.serp", "SerpSearchEngine"),
    "RerankerConfig": (".internal.servers.retrieval.rerank", "RerankerConfig"),
    "create_base_app": (".internal.servers.app", "create_base_app"),
    "create_search_app": (".internal.servers.app", "create_search_app"),
    "format_document": (".internal.servers.app", "format_document"),
    # backend.document_index.index_builder
    "IndexBuilder": (".internal.document_index.index_builder", "IndexBuilder"),
    "IndexBuilderConfig": (
        ".internal.document_index.index_builder",
        "IndexBuilderConfig",
    ),
    "prepare_texts": (".internal.document_index.index_builder", "prepare_texts"),
    "resolve_pooling_method": (
        ".internal.document_index.index_builder",
        "resolve_pooling_method",
    ),
    "pooling": (".internal.document_index.index_builder", "pooling"),
    "set_hnsw_ef_construction": (
        ".internal.document_index.index_builder",
        "set_hnsw_ef_construction",
    ),
    "set_hnsw_ef_search": (
        ".internal.document_index.index_builder",
        "set_hnsw_ef_search",
    ),
    # backend.document_index.retrieval
    "DenseRetriever": (".internal.document_index.retrieval", "DenseRetriever"),
    "DenseRetrieverConfig": (
        ".internal.document_index.retrieval",
        "DenseRetrieverConfig",
    ),
    # internal.servers.retrieval.rerank
    "SentenceTransformerReranker": (
        ".internal.servers.retrieval.rerank",
        "SentenceTransformerReranker",
    ),
    "get_reranker": (".internal.servers.retrieval.rerank", "get_reranker"),
    "passage_to_string": (".internal.servers.retrieval.rerank", "passage_to_string"),
    "string_to_document": (".internal.servers.retrieval.rerank", "string_to_document"),
    # backend.document_index.text
    "SOS_token": (".internal.document_index.text", "SOS_token"),
    "EOS_token": (".internal.document_index.text", "EOS_token"),
    "MAX_LENGTH": (".internal.document_index.text", "MAX_LENGTH"),
    "normalize_text": (".internal.document_index.text", "normalize_text"),
    "normalize_document": (".internal.document_index.text", "normalize_document"),
    "tokenize_text": (".internal.document_index.text", "tokenize_text"),
    "tokenize_document": (".internal.document_index.text", "tokenize_document"),
    "build_vocabulary_from_sequences": (
        ".internal.document_index.text",
        "build_vocabulary_from_sequences",
    ),
    "extract_keywords": (".internal.document_index.text", "extract_keywords"),
    # context
    "AgentBehavior": (".context", "AgentBehavior"),
    "AgentBehaviorConfig": (".context", "AgentBehaviorConfig"),
    "AnswerGenerationRequest": (".context", "AnswerGenerationRequest"),
    "AnswerGenerationResult": (".context", "AnswerGenerationResult"),
    "AnswerStyle": (".context", "AnswerStyle"),
    "ChatMessage": (".context", "ChatMessage"),
    "ContextDocument": (".context", "ContextDocument"),
    "ContextSection": (".context", "ContextSection"),
    "EvidenceSnippet": (".context", "EvidenceSnippet"),
    "LLMResponse": (".context", "LLMResponse"),
    "PromptBundle": (".context", "PromptBundle"),
    "SearchContextBundle": (".context", "SearchContextBundle"),
    "SearchFilters": (".context", "SearchFilters"),
    "SearchRequest": (".context", "SearchRequest"),
    "SearchType": (".context", "SearchType"),
    "answer_with_retrieval": (".context", "answer_with_retrieval"),
    "build_answer_prompt": (".context", "build_answer_prompt"),
    "build_chat_prompt": (".context", "build_chat_prompt"),
    "build_context_bundle": (".context", "build_context_bundle"),
    "build_retrieval_prompt": (".context", "build_retrieval_prompt"),
    "extract_citations": (".context", "extract_citations"),
    "generate_answer": (".context", "generate_answer"),
    "rank_evidence_snippets": (".context", "rank_evidence_snippets"),
    "retrieve_context": (".context", "retrieve_context"),
    "synthesize_answer_from_context": (
        ".context",
        "synthesize_answer_from_context",
    ),
    # feature_flags
    "EnvFeatureFlagProvider": (".internal.feature_flags", "EnvFeatureFlagProvider"),
    "NoOpFeatureFlagProvider": (".internal.feature_flags", "NoOpFeatureFlagProvider"),
    "StaticFeatureFlagProvider": (
        ".internal.feature_flags",
        "StaticFeatureFlagProvider",
    ),
    "get_feature_flag_provider": (
        ".internal.feature_flags",
        "get_feature_flag_provider",
    ),
    "is_feature_enabled": (".internal.feature_flags", "is_feature_enabled"),
    # hooks
    "HookConfig": (".internal.hooks", "HookConfig"),
    "HookExecutionError": (".internal.hooks", "HookExecutionError"),
    "HookFailStrategy": (".internal.hooks", "HookFailStrategy"),
    "HookPoint": (".internal.hooks", "HookPoint"),
    "HookRegistry": (".internal.hooks", "HookRegistry"),
    "HookSkipped": (".internal.hooks", "HookSkipped"),
    "HookSoftFailed": (".internal.hooks", "HookSoftFailed"),
    "execute_hook": (".internal.hooks", "execute_hook"),
    # internal.db
    "AgenticSearchStore": (".internal.db", "AgenticSearchStore"),
    "ChatMessageRecord": (".internal.db", "ChatMessageRecord"),
    "ChatSessionRecord": (".internal.db", "ChatSessionRecord"),
    "ConnectorConfig": (".internal.db", "ConnectorConfig"),
    "DocumentPermission": (".internal.db", "DocumentPermission"),
    "GroupRecord": (".internal.db", "GroupRecord"),
    "IndexAttemptRecord": (".internal.db", "IndexAttemptRecord"),
    "StoredDocument": (".internal.db", "StoredDocument"),
    "UserRecord": (".internal.db", "UserRecord"),
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
