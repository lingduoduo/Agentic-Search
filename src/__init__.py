"""Top-level package for Agentic-Search."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OnlineSearchConfig",
    "OnlineSearchEngine",
    "SerpSearchConfig",
    "SerpSearchEngine",
    "IndexBuilder",
    "IndexBuilderConfig",
    "DenseRetriever",
    "DenseRetrieverConfig",
    "RetrievalServerConfig",
    "RerankerConfig",
    "RetrievalRerankConfig",
    "create_base_app",
    "Vocabulary",
    "SOS_token",
    "EOS_token",
    "MAX_LENGTH",
    "normalize_text",
    "tokenize_text",
    "build_vocabulary_from_sequences",
    "extract_keywords",
    "ActorRolloutStep",
    "ContinuationDecision",
    "FinalGenBatchOutput",
    "GenerationConfig",
    "GRPOPromptGroupResult",
    "GRPOTrainingStepResult",
    "GroupedRolloutBatch",
    "LLMGenerationManager",
    "PPOPolicyLossConfig",
    "ReActContextTransition",
    "SearchDecision",
    "SearchBatch",
    "SearchStep",
    "SearchTrajectoryLog",
    "TensorConfig",
    "TensorHelper",
    "assign_group_relative_advantages",
    "format_search_trajectory_log",
    "AgentLoopBase",
    "AgentLoopConfig",
    "AgentLoopOutput",
    "GRPOAdvantageConfig",
    "GRPORolloutSample",
    "PromptGroupSamplingConfig",
    "ScoredGRPORollout",
    "SFTExample",
    "SingleTurnAgentLoop",
    "build_grpo_sampling_params",
    "compute_grpo_outcome_advantage",
    "build_search_sft_example",
    "get_registered_agent_loop",
    "list_registered_agent_loops",
    "sample_prompt_group",
    "score_prompt_group",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ActorRolloutStep",
        "ContinuationDecision",
        "FinalGenBatchOutput",
        "GenerationConfig",
        "GRPOPromptGroupResult",
        "GRPOTrainingStepResult",
        "GroupedRolloutBatch",
        "LLMGenerationManager",
        "PPOPolicyLossConfig",
        "ReActContextTransition",
        "SearchDecision",
        "SearchBatch",
        "SearchStep",
        "SearchTrajectoryLog",
        "TensorConfig",
        "TensorHelper",
        "assign_group_relative_advantages",
        "format_search_trajectory_log",
    }:
        llm_agent_module = import_module(".llm_agent", __name__)
        return getattr(llm_agent_module, name)
    if name in {
        "AgentLoopBase",
        "AgentLoopConfig",
        "AgentLoopOutput",
        "GRPOAdvantageConfig",
        "GRPORolloutSample",
        "PromptGroupSamplingConfig",
        "ScoredGRPORollout",
        "SFTExample",
        "SingleTurnAgentLoop",
        "build_grpo_sampling_params",
        "compute_grpo_outcome_advantage",
        "build_search_sft_example",
        "get_registered_agent_loop",
        "list_registered_agent_loops",
        "sample_prompt_group",
        "score_prompt_group",
    }:
        agent_loop_module = import_module(".agent_loop", __name__)
        return getattr(agent_loop_module, name)
    if name in __all__:
        search_module = import_module(".search", __name__)
        return getattr(search_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
