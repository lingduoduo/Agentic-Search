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
from .backend.connectors import BaseConnector as BaseConnector
from .backend.connectors import CheckpointedConnector as CheckpointedConnector
from .backend.connectors import (
    CheckpointedConnectorWithPermSync as CheckpointedConnectorWithPermSync,
)
from .backend.connectors import ConnectorCheckpoint as ConnectorCheckpoint
from .backend.connectors import ConnectorFailure as ConnectorFailure
from .backend.connectors import CredentialsConnector as CredentialsConnector
from .backend.connectors import (
    CredentialsProviderInterface as CredentialsProviderInterface,
)
from .backend.connectors import Document as Document
from .backend.connectors import EventConnector as EventConnector
from .backend.connectors import HierarchyConnector as HierarchyConnector
from .backend.connectors import HierarchyNode as HierarchyNode
from .backend.connectors import InMemoryConnector as InMemoryConnector
from .backend.connectors import LoadConnector as LoadConnector
from .backend.connectors import LocalFileConnector as LocalFileConnector
from .backend.connectors import LocalFilePollConnector as LocalFilePollConnector
from .backend.connectors import LocalFileSlimConnector as LocalFileSlimConnector
from .backend.connectors import (
    LocalFileSlimConnectorWithPermSync as LocalFileSlimConnectorWithPermSync,
)
from .backend.connectors import OAuthConnector as OAuthConnector
from .backend.connectors import PollConnector as PollConnector
from .backend.connectors import Resolver as Resolver
from .backend.connectors import SearchConnector as SearchConnector
from .backend.connectors import SlimConnector as SlimConnector
from .backend.connectors import SlimConnectorWithPermSync as SlimConnectorWithPermSync
from .backend.connectors import SlimDocument as SlimDocument
from .backend.connectors import StaticCredentialsProvider as StaticCredentialsProvider
from .backend.connectors import batched as batched
from .backend.db import AgenticSearchStore as AgenticSearchStore
from .backend.db import ChatMessageRecord as ChatMessageRecord
from .backend.db import ChatSessionRecord as ChatSessionRecord
from .backend.db import ConnectorConfig as ConnectorConfig
from .backend.db import DocumentPermission as DocumentPermission
from .backend.db import GroupRecord as GroupRecord
from .backend.db import IndexAttemptRecord as IndexAttemptRecord
from .backend.db import StoredDocument as StoredDocument
from .backend.db import UserRecord as UserRecord
from .context.retrieval.client import SearchClient as SearchClient
from .context.retrieval.client import SearchClientConfig as SearchClientConfig
from .context.search import AgentContext as AgentContext
from .context.search import SearchContext as SearchContext
from .context.search import SearchResult as SearchResult
from .backend.document_index.retrieval import SparseRetriever as SparseRetriever
from .backend.document_index.retrieval import (
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
    # servers
    "OnlineSearchConfig": (".backend.servers.google", "OnlineSearchConfig"),
    "OnlineSearchEngine": (".backend.servers.google", "OnlineSearchEngine"),
    "SerpSearchConfig": (".backend.servers.serp", "SerpSearchConfig"),
    "SerpSearchEngine": (".backend.servers.serp", "SerpSearchEngine"),
    "RetrievalServerConfig": (
        ".backend.servers.retrieval_server",
        "RetrievalServerConfig",
    ),
    "RetrievalRerankConfig": (
        ".backend.servers.retrieval_rerank",
        "RetrievalRerankConfig",
    ),
    "RerankerConfig": (".backend.servers.rerank", "RerankerConfig"),
    "create_base_app": (".backend.servers.app", "create_base_app"),
    "create_search_app": (".backend.servers.app", "create_search_app"),
    "format_document": (".backend.servers.app", "format_document"),
    # backend.document_index.index_builder
    "IndexBuilder": (".backend.document_index.index_builder", "IndexBuilder"),
    "IndexBuilderConfig": (
        ".backend.document_index.index_builder",
        "IndexBuilderConfig",
    ),
    "prepare_texts": (".backend.document_index.index_builder", "prepare_texts"),
    "resolve_pooling_method": (
        ".backend.document_index.index_builder",
        "resolve_pooling_method",
    ),
    "pooling": (".backend.document_index.index_builder", "pooling"),
    "set_hnsw_ef_construction": (
        ".backend.document_index.index_builder",
        "set_hnsw_ef_construction",
    ),
    "set_hnsw_ef_search": (
        ".backend.document_index.index_builder",
        "set_hnsw_ef_search",
    ),
    # backend.document_index.retrieval
    "DenseRetriever": (".backend.document_index.retrieval", "DenseRetriever"),
    "DenseRetrieverConfig": (
        ".backend.document_index.retrieval",
        "DenseRetrieverConfig",
    ),
    # backend.servers.rerank
    "SentenceTransformerReranker": (
        ".backend.servers.rerank",
        "SentenceTransformerReranker",
    ),
    "get_reranker": (".backend.servers.rerank", "get_reranker"),
    "passage_to_string": (".backend.servers.rerank", "passage_to_string"),
    "string_to_document": (".backend.servers.rerank", "string_to_document"),
    # backend.document_index.text
    "SOS_token": (".backend.document_index.text", "SOS_token"),
    "EOS_token": (".backend.document_index.text", "EOS_token"),
    "MAX_LENGTH": (".backend.document_index.text", "MAX_LENGTH"),
    "normalize_text": (".backend.document_index.text", "normalize_text"),
    "normalize_document": (".backend.document_index.text", "normalize_document"),
    "tokenize_text": (".backend.document_index.text", "tokenize_text"),
    "tokenize_document": (".backend.document_index.text", "tokenize_document"),
    "build_vocabulary_from_sequences": (
        ".backend.document_index.text",
        "build_vocabulary_from_sequences",
    ),
    "extract_keywords": (".backend.document_index.text", "extract_keywords"),
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
    "EnvFeatureFlagProvider": (".backend.feature_flags", "EnvFeatureFlagProvider"),
    "NoOpFeatureFlagProvider": (".backend.feature_flags", "NoOpFeatureFlagProvider"),
    "StaticFeatureFlagProvider": (
        ".backend.feature_flags",
        "StaticFeatureFlagProvider",
    ),
    "get_feature_flag_provider": (
        ".backend.feature_flags",
        "get_feature_flag_provider",
    ),
    "is_feature_enabled": (".backend.feature_flags", "is_feature_enabled"),
    # hooks
    "HookConfig": (".backend.hooks", "HookConfig"),
    "HookExecutionError": (".backend.hooks", "HookExecutionError"),
    "HookFailStrategy": (".backend.hooks", "HookFailStrategy"),
    "HookPoint": (".backend.hooks", "HookPoint"),
    "HookRegistry": (".backend.hooks", "HookRegistry"),
    "HookSkipped": (".backend.hooks", "HookSkipped"),
    "HookSoftFailed": (".backend.hooks", "HookSoftFailed"),
    "execute_hook": (".backend.hooks", "execute_hook"),
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
