"""Post-training and fine-tuning for the search agent.

Shared building blocks live at this level -- ``data`` (prompt and dataset
construction), ``reward`` (reward functions) and ``judge`` (RLAIF judges) are
used by more than one method. Each method gets its own package: ``sft`` for
supervised fine-tuning, ``dpo`` for Direct Preference Optimization over
preference pairs, ``rl`` for the GRPO/PPO stack, and ``eval`` for the benchmark
harnesses.

``train_query_router`` is the odd one out: an offline scikit-learn trainer for
the QueryRouter, not LLM post-training.
"""

try:
    from .rl.rollouts import GRPOAdvantageConfig as GRPOAdvantageConfig
    from .rl.rollouts import PromptGroupSamplingConfig as PromptGroupSamplingConfig
    from .rl.rollouts import compute_dapo_advantages as compute_dapo_advantages
    from .rl.rollouts import (
        compute_grpo_outcome_advantage as compute_grpo_outcome_advantage,
    )
    from .rl.rollouts import score_prompt_group as score_prompt_group
    from .rl import PPORewardManager as PPORewardManager
    from .rl import compute_grpo_policy_loss as compute_grpo_policy_loss
    from .reward import BatchJudgeFn as BatchJudgeFn
    from .reward import CompositeRewardConfig as CompositeRewardConfig
    from .reward import JudgeFn as JudgeFn
    from .reward import SearchRewardFunction as SearchRewardFunction
    from .reward import format_compliance_reward as format_compliance_reward
    from .reward import token_f1_score as token_f1_score
    from .sft.trainer import SFTExample as SFTExample
    from .sft.trainer import build_search_sft_example as build_search_sft_example
    from .judge import GoldAgreementJudge as GoldAgreementJudge
    from .judge import LLMJudge as LLMJudge
    from .judge import SimulatedPreferenceJudge as SimulatedPreferenceJudge
    from .judge import judge_gold_agreement as judge_gold_agreement
    from .data import PromptBatch as PromptBatch
    from .data import PromptTrainingExample as PromptTrainingExample
    from .data import build_search_rag_record as build_search_rag_record
    from .data import build_search_qa_messages as build_search_qa_messages
    from .data import build_search_qa_record as build_search_qa_record
    from .data import build_search_qa_prompt as build_search_qa_prompt
    from .data import format_rag_reference as format_rag_reference
    from .data import make_search_rag_map_fn as make_search_rag_map_fn
    from .data import make_search_qa_map_fn as make_search_qa_map_fn
    from .data import normalize_question_text as normalize_question_text
    from .rl.qlearning import QLearningAgent as QLearningAgent
    from .rl.search_environment import SearchEnvironment as SearchEnvironment
except ImportError:
    pass
