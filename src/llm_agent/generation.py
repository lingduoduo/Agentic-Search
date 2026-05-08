"""Generation manager for search-oriented, tool-using LLM loops."""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    from src.agent_loop.agent_loop import AgentLoopOutput

import torch

from .tensor_helper import TensorConfig, TensorHelper

logger = logging.getLogger(__name__)


@dataclass
class SearchBatch:
    """Lightweight batch container for multi-turn search generation."""

    batch: dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_batch: dict[str, Any] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, batch: dict[str, torch.Tensor]) -> "SearchBatch":
        return cls(batch=batch, non_tensor_batch={}, meta_info={})


@dataclass
class AgentLoopState:
    """Mutable state for the multi-turn generate -> act -> observe loop."""

    rollings: SearchBatch
    original_left_side: dict[str, torch.Tensor]
    original_right_side: dict[str, torch.Tensor]
    active_mask: torch.Tensor
    trajectory_turns: list[int]
    turns_stats: torch.Tensor
    valid_action_stats: torch.Tensor
    valid_search_stats: torch.Tensor
    active_num_list: list[int]
    meta_info: dict[str, Any] = field(default_factory=dict)
    # Per-rollout cache: (batch_index, query) -> retrieval result string.
    # Prevents re-calling the retriever when the policy repeats a query within
    # the same trajectory.  Keyed by (batch_index, query) so different
    # trajectories never share cached results.
    search_cache: dict[tuple[int, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentLoopStepResult:
    """Outcome of one orchestration step for the active trajectories."""

    responses_ids: torch.Tensor
    responses_str: list[str]
    parsed_actions: list["PolicyAction"]
    next_obs: list[str]
    dones: list[int]
    valid_action: list[int]
    is_search: list[int]


@dataclass(frozen=True)
class ActorRolloutStep:
    """One policy rollout step before tool execution or environment feedback.

    In RL terms, this is the sampled action emitted by the current policy for
    the current state. In this agent setting the action space is expressed as
    structured text such as `<search>`, `<plan>`, `<fetch>`, or `<answer>`.
    """

    responses_ids: torch.Tensor
    responses_str: list[str]
    parsed_actions: list["PolicyAction"]


@dataclass(frozen=True)
class PolicyAction:
    """One action sampled by the policy from the current state."""

    tag: str | None
    content: str
    raw_text: str


@dataclass(frozen=True)
class SearchToolCall:
    """One model-chosen search invocation for the current rollout state."""

    query: str
    problem: str
    ground_truth: str
    batch_index: int


@dataclass(frozen=True)
class SearchDecision:
    """One explicit model decision to call the search tool.

    This captures the critical RL step where the policy decides:
    - whether to search at all
    - what query to issue
    - which trajectory in the batch requested it
    """

    query: str
    batch_index: int
    problem: str
    ground_truth: str
    raw_action_text: str


@dataclass(frozen=True)
class FetchToolCall:
    """One model-chosen page fetch invocation for the current rollout state."""

    urls: list[str]
    batch_index: int


@dataclass(frozen=True)
class ReActStep:
    """One (action, observation) pair in a ReAct trajectory.

    Captures the full context update for one turn:
        context += action_text     # model output
        context += observation     # <information>...</information> injected back

    Accumulating these across turns gives the trajectory the model conditions
    on during the next generation step.
    """

    turn: int
    action_tag: str | None
    action_content: str
    observation: str
    is_terminal: bool


@dataclass(frozen=True)
class ReActContextTransition:
    """One context update fed back into the next model state.

    ReAct-style agent loops do not just keep the model action. They append:

        context += model_output
        context += environment_observation

    so the next generation step conditions on both the chosen action and the
    retrieved information or tool feedback that came back.
    """

    action_text: str
    observation_text: str
    appended_context_text: str


@dataclass(frozen=True)
class ContinuationDecision:
    """Whether the agent should take another generation turn.

    This makes second-round and later-round behavior explicit: if the policy
    has not emitted `<answer>`, the loop can continue with another search,
    fetch, or answer step using the updated ReAct context.
    """

    turn_index: int
    continue_generation: bool
    active_trajectories: int
    reason: str


@dataclass(frozen=True)
class SearchStep:
    """One fully logged agent step for RL trajectory inspection."""

    step_id: int
    state: str
    model_output: str
    action_type: str
    action_value: str
    observation: str | None
    done: bool = False


@dataclass(frozen=True)
class SearchTrajectoryLog:
    """Human-readable trajectory log for one prompt.

    This is the primary RL data record: everything that happened during one
    agent rollout, structured for printing, saving, and reward labelling.

    Usage::

        log = trajectory_logs[0]
        print(log)                                     # compact trace
        print(log.to_dict())                           # dict for JSONL
        print(format_search_trajectory_log(log,
              reward=1.0, verbose=True))               # full debug dump
    """

    batch_index: int
    question: str
    steps: list[SearchStep]
    final_answer: str | None
    finished_without_answer: bool

    def __str__(self) -> str:
        return format_search_trajectory_log(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSONL saving and RL data export."""
        return {
            "batch_index": self.batch_index,
            "question": self.question,
            "steps": [
                {
                    "step_id": step.step_id,
                    "action_type": step.action_type,
                    "action_value": step.action_value,
                    "observation": step.observation,
                    "done": step.done,
                }
                for step in self.steps
            ],
            "final_answer": self.final_answer,
            "finished_without_answer": self.finished_without_answer,
        }


@dataclass(frozen=True)
class RolloutTrajectory:
    """One complete RL rollout trajectory for a single prompt."""

    batch_index: int
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_with_observation_mask: list[int]
    trajectory_turns: int
    steps: list[ReActStep]
    final_answer: str | None
    finished_without_answer: bool
    tokens: list[int] | None = None
    attention_mask: list[int] | None = None
    response_mask: list[int] | None = None
    old_log_probs: list[float] | None = None
    new_log_probs: list[float] | None = None


@dataclass(frozen=True)
class FinalGenBatchOutput:
    """Final rollout artifact for one generated batch.

    This is the RL rollout object after the full agent loop finishes.  It
    consolidates everything a training loop needs in one place:

        prompt → search → retrieval → reasoning → answer
              ↓
        search_batch      — tensor layout for policy/log-prob/loss computation
        trajectories      — one RolloutTrajectory per prompt (symbolic trace)
        rollout_outputs   — one AgentLoopOutput per trajectory (reward interface)
        trajectory_turns  — which turn each trajectory completed at
                            (0 = never answered, 1..max_turns = answered in turn N,
                             max_turns+1 = forced answer after budget exhaustion)

    Intended usage::

        final = manager.build_final_gen_batch_output(batch)
        rewards = reward_fn.compute_batch(final.rollout_outputs, ground_truths, judge)
        advantages = reward_fn.assign_grpo_outcome_token_advantages(
            final.rollout_outputs, ground_truths, judge, group_ids
        )
    """

    search_batch: SearchBatch
    trajectories: list["RolloutTrajectory"]
    trajectory_logs: list["SearchTrajectoryLog"]
    rollout_outputs: list["AgentLoopOutput"]
    trajectory_turns: list[int]


@dataclass(frozen=True)
class GroupedRolloutBatch:
    """One rollout sampled as part of a GRPO prompt group."""

    group_id: str
    rollout_index: int
    sampling_params: dict[str, Any]
    final_output: FinalGenBatchOutput


@dataclass(frozen=True)
class ScoredGroupedRollout:
    """A GroupedRolloutBatch with its scalar reward and within-group GRPO advantage.

    Produced by :func:`score_group_rollout`.  Contains everything a GRPO
    training step needs for one rollout: the full trajectory output for
    policy-gradient updates, the scalar reward for logging, and the
    group-normalized advantage for the surrogate objective.

    Typical flow::

        grouped = manager.run_prompt_rollout_group(prompt_batch, ...)
        scored  = score_group_rollout(grouped,
                      ground_truth="...", judge_fn=exact_match)
        print(format_group_rollout(grouped,
                  rewards=[s.reward for s in scored],
                  advantages=[s.advantage for s in scored]))
    """

    group_id: str
    rollout_index: int
    sampling_params: dict[str, Any]
    final_output: FinalGenBatchOutput
    reward: float
    reward_components: dict[str, float]
    advantage: float


@dataclass(frozen=True)
class RetrievedDocument:
    """Normalized retrieval document across BM25, dense, hybrid, web, and RAG backends."""

    title: str
    snippet: str
    url: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class PPOPolicyLossConfig:
    """Configuration for PPO / GRPO clipped policy optimization."""

    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0
    # Coefficient for the entropy bonus added to the objective.  A positive
    # value encourages the search / reasoning / stopping policies to remain
    # diverse and prevents premature mode collapse onto a single strategy.
    # Requires new_log_probs to be present in batch.batch.
    entropy_coefficient: float = 0.0
    # Per-action-type loss multipliers.  Keys: "search", "plan", "fetch", "answer".
    # Tokens with no matching key keep weight 1.0.  None means uniform weighting.
    action_type_weights: dict[str, float] | None = None


ROLLOUT_ACTION_TAGS = ("plan", "search", "fetch", "answer")
ACTION_PATTERN = re.compile(
    rf"<(?P<tag>{'|'.join(ROLLOUT_ACTION_TAGS)})>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL,
)

_NO_INFO = "No information available"

# Clamp applied to log(π_θ / π_θ_old) before torch.exp() in prob_ratio and KL
# computations.  Float32 can represent exp(88) ≈ 1.7e38 before overflow; 20.0
# caps the ratio at ~4.9e8 while keeping gradients numerically stable in early
# training when the policy can diverge quickly.
_LOG_RATIO_CLAMP = 20.0


def _token_char_offsets(ids: list[int], tokenizer: Any) -> list[tuple[int, int]]:
    """Return (char_start, char_end) for each token in the decoded response.

    Tries HF-style ``return_offsets_mapping`` first for accuracy with BPE /
    SentencePiece tokenizers.  Falls back to cumulative per-token decode for
    character-level and test tokenizers that don't support the HF calling
    convention.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False)
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        if len(offsets) == len(ids):
            return list(offsets)
    except Exception:
        pass
    pos = 0
    offsets_fallback: list[tuple[int, int]] = []
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        offsets_fallback.append((pos, pos + len(piece)))
        pos += len(piece)
    return offsets_fallback


# ---------------------------------------------------------------------------
# Retriever protocol and backends
# ---------------------------------------------------------------------------


def _parse_api_item(item: dict[str, Any]) -> RetrievedDocument:
    """Normalize one API result item from any supported backend shape."""
    document = item.get("document", item) if isinstance(item, dict) else {}
    score = item.get("score") if isinstance(item, dict) else None

    title = snippet = ""
    url = None

    if isinstance(document, dict):
        title = str(document.get("title") or "").strip()
        url = document.get("url")
        raw_contents = document.get("contents")
        raw_snippet = document.get("snippet")
        if isinstance(raw_contents, str) and raw_contents.strip():
            first_line, sep, remainder = raw_contents.partition("\n")
            normalized_title = first_line.strip().strip('"')
            if not title and normalized_title:
                title = normalized_title
            snippet = remainder.strip() if sep else str(raw_snippet or "").strip()
        else:
            snippet = str(raw_snippet or "").strip()
    else:
        snippet = str(document).strip()

    if isinstance(item, dict):
        if not title:
            title = str(item.get("title") or "").strip()
        if not snippet:
            snippet = str(item.get("snippet") or item.get("contents") or "").strip()
        if url is None:
            raw_url = item.get("url") or item.get("link")
            url = str(raw_url).strip() if raw_url else None

    return RetrievedDocument(
        title=title or "No title.",
        snippet=snippet or "No snippet available.",
        url=url,
        score=float(score)
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else None,
    )


def _normalize_retrieve_result_rows(
    payload: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize `/retrieve` payloads from local, web, and RAG backends.

    Common accepted shapes:
    - {"result": [[{...}, {...}]]}     # POST /retrieve single-query batch
    - {"result": [{...}, {...}]}       # looser single-query variant
    - {"results": [{...}, {...}]}      # standalone single-query retrieval service
    - [{...}, {...}]                   # direct rows
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    result = payload.get("result", payload.get("results", []))
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            rows = result[0]
        else:
            rows = result
        return [item for item in rows if isinstance(item, dict)]
    return []


def _documents_from_retrieve_payload(
    payload: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[RetrievedDocument]:
    """Parse a retrieval payload into normalized document objects."""
    return [_parse_api_item(item) for item in _normalize_retrieve_result_rows(payload)]


def _documents_per_query_from_payload(
    payload: dict[str, Any],
    n_queries: int,
) -> list[list[RetrievedDocument]]:
    """Parse a multi-query /retrieve response into per-query document lists.

    Expected shape from the endpoint::

        {"result": [[doc, doc, ...], [doc, doc, ...], ...]}
        {"results": [[doc, doc, ...], [doc, doc, ...], ...]}

    where ``result[i]`` contains documents for query ``i``.  Also handles the
    single-query flat variant ``{"result": [doc, ...]}`` by returning it as the
    first entry and empty lists for remaining queries.
    """
    result = (
        payload.get("result", payload.get("results", []))
        if isinstance(payload, dict)
        else []
    )
    if not isinstance(result, list) or not result:
        return [[] for _ in range(n_queries)]
    if isinstance(result[0], list):
        per_query = [
            [_parse_api_item(item) for item in rows if isinstance(item, dict)]
            for rows in result[:n_queries]
        ]
        per_query += [[] for _ in range(n_queries - len(per_query))]
        return per_query
    # Flat list — treat as single-query response
    docs = [_parse_api_item(item) for item in result if isinstance(item, dict)]
    return [docs] + [[] for _ in range(n_queries - 1)]


def build_react_observation(action: "PolicyAction", retrieval_result: str = "") -> str:
    """Build the observation injected back into context after the model's action.

    Completes the ReAct loop for one turn:
        model generates action → environment returns observation → model sees it next turn

    For <search>: wraps retrieval result in <information>...</information>.
    For <answer>: returns "" (terminal — no observation needed).
    For other tags: returns structured feedback the model can act on.
    """
    if action.tag == "search":
        content = retrieval_result.strip() or _NO_INFO
        return f"\n\n<information>{content}</information>\n\n"
    if action.tag == "plan":
        return (
            "\n\n<plan_feedback>Plan recorded. Continue with <search>, <fetch>, or "
            "<answer>.</plan_feedback>\n\n"
        )
    if action.tag == "fetch":
        content = retrieval_result.strip() or _NO_INFO
        return f"\n\n<full_page>{content}</full_page>\n\n"
    if action.tag == "answer":
        return ""
    return (
        "\nMy previous action is invalid. "
        "Valid actions in this rollout loop are <plan>, <search>, <fetch>, and "
        "<answer>. Let me try again.\n"
    )


def format_search_trajectory_log(
    trajectory_log: "SearchTrajectoryLog",
    *,
    reward: float | None = None,
    verbose: bool = False,
    max_obs_chars: int = 300,
) -> str:
    """Render one logged trajectory as a printable text trace.

    Compact mode (default) produces the minimal RL-readable trace::

        Question: Who won the 2024 Nobel Prize in Physics?
        Step 1: search 2024 Nobel Prize Physics winner
          Observation: The 2024 Nobel Prize was awarded to Hopfield and Hinton ...
        Step 2: answer John Hopfield and Geoffrey Hinton.
        Final Answer: John Hopfield and Geoffrey Hinton.
        Reward: 1.0

    Verbose mode (``verbose=True``) additionally includes the full state
    context and raw model output for each step — useful for debugging.
    """
    lines = [f"Question: {trajectory_log.question}"]
    for step in trajectory_log.steps:
        lines.append(
            f"Step {step.step_id}: {step.action_type} {step.action_value}".rstrip()
        )
        if verbose:
            lines.append(f"  State: {step.state}")
            lines.append(f"  Model Output: {step.model_output}")
        if step.observation:
            obs = step.observation
            if not verbose and len(obs) > max_obs_chars:
                obs = obs[:max_obs_chars] + "..."
            lines.append(f"  Observation: {obs}")
    if trajectory_log.final_answer is not None:
        lines.append(f"Final Answer: {trajectory_log.final_answer}")
    if reward is not None:
        lines.append(f"Reward: {reward}")
    return "\n".join(lines)


def format_trajectory_batch(
    logs: "list[SearchTrajectoryLog]",
    rewards: "list[float] | None" = None,
    *,
    verbose: bool = False,
    max_obs_chars: int = 300,
) -> str:
    """Format all trajectories from one rollout batch as a single printable string.

    Each trajectory is separated by a horizontal rule so you can scan through
    the batch output to see what the agent did for each prompt.  Call after
    ``run_llm_loop`` / ``build_final_gen_batch_output`` to inspect a training
    batch before computing rewards::

        batch, _ = manager.run_llm_loop(...)
        logs = batch.non_tensor_batch["trajectory_logs"]
        rewards = [reward_fn.compute(o, gt, judge) for o, gt in zip(outputs, gts)]
        print(format_trajectory_batch(logs, rewards))
    """
    separator = "\n" + "─" * 60 + "\n"
    parts = [
        format_search_trajectory_log(
            log,
            reward=rewards[i] if rewards is not None and i < len(rewards) else None,
            verbose=verbose,
            max_obs_chars=max_obs_chars,
        )
        for i, log in enumerate(logs)
    ]
    return separator.join(parts)


def score_group_rollout(
    grouped_rollouts: "list[GroupedRolloutBatch]",
    *,
    ground_truth: str,
    judge_fn: "Callable[[str, str], float]",
    reward_fn: Any = None,
    advantage_config: Any = None,
    batch_judge_fn: Any = None,
) -> "list[ScoredGroupedRollout]":
    """Score a GRPO prompt group from :meth:`LLMGenerationManager.run_prompt_rollout_group`.

    Bridges the ``LLMGenerationManager``-based rollout output to the standard
    GRPO scoring pipeline:

        run_prompt_rollout_group()        → list[GroupedRolloutBatch]
        score_group_rollout(...)          → list[ScoredGroupedRollout]
        format_group_rollout(...)         → printable group trace

    Each rollout's ``rollout_outputs[0]`` is the ``AgentLoopOutput`` for the
    (single-prompt) trajectory.  Multi-prompt batches keep only the first
    output per rollout — for multi-prompt groups use :func:`score_prompt_batch`
    from ``src.agent_loop.grpo`` directly.

    Args:
        grouped_rollouts: List returned by ``run_prompt_rollout_group``.
        ground_truth:     Reference answer for the shared prompt.
        judge_fn:         ``(answer, ground_truth) -> float`` in ``[0, 1]``.
        reward_fn:        Optional :class:`SearchRewardFunction` instance.
        advantage_config: Optional :class:`GRPOAdvantageConfig`.
        batch_judge_fn:   Optional batch judge for LLM-based scoring.
    """
    from src.agent_loop.grpo import GRPORolloutSample, score_prompt_group
    from src.agent_loop.reward import SearchRewardFunction

    if not grouped_rollouts:
        return []

    samples = [
        GRPORolloutSample(
            group_id=grb.group_id,
            rollout_index=grb.rollout_index,
            sampling_params=grb.sampling_params,
            output=grb.final_output.rollout_outputs[0],
        )
        for grb in grouped_rollouts
        if grb.final_output.rollout_outputs
    ]
    if not samples:
        return []

    scored = score_prompt_group(
        samples,
        ground_truth=ground_truth,
        judge_fn=judge_fn,
        reward_fn=reward_fn or SearchRewardFunction(),
        advantage_config=advantage_config,
        batch_judge_fn=batch_judge_fn,
    )
    scored_by_index = {s.rollout_index: s for s in scored}
    return [
        ScoredGroupedRollout(
            group_id=grb.group_id,
            rollout_index=grb.rollout_index,
            sampling_params=grb.sampling_params,
            final_output=grb.final_output,
            reward=scored_by_index[grb.rollout_index].reward,
            reward_components=scored_by_index[grb.rollout_index].reward_components,
            advantage=scored_by_index[grb.rollout_index].advantage,
        )
        for grb in grouped_rollouts
        if grb.rollout_index in scored_by_index
    ]


def _grpo_advantages(rewards: list[float], *, normalize: bool) -> list[float]:
    """Mean-center rewards; optionally divide by within-group std.

    Single-sample groups get advantage 0.0 — no relative comparison exists.
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if not normalize:
        return centered
    variance = sum(c * c for c in centered) / n
    std = math.sqrt(variance)
    return [c / (std + 1e-8) for c in centered]


def assign_group_relative_advantages(
    grouped_rollouts: "list[GroupedRolloutBatch]",
    *,
    rewards: "list[float]",
    reward_components: "list[dict[str, float]] | None" = None,
    normalize: bool = True,
) -> "list[ScoredGroupedRollout]":
    """Attach GRPO outcome advantages to one prompt group's rollout trajectories.

    Applies the group-relative GRPO advantage transform to pre-computed scalar
    rewards so you can use any reward function without going through the full
    ``score_group_rollout`` pipeline:

        grouped = manager.run_prompt_rollout_group(prompt_batch, ...)
        rewards = [1.0, 0.7, 0.0, 0.0]
        scored  = assign_group_relative_advantages(grouped, rewards=rewards)
        print(format_scored_group_rollout(scored))

    With ``normalize=True`` (default) advantages are std-normalized within the
    group — the GRPO objective is more stable this way:

        advantage_i = (reward_i − mean) / (std(group) + ε)

    With ``normalize=False`` only mean-centering is applied (the original GRPO
    paper formula):

        advantage_i = reward_i − mean(group_rewards)

    In both modes a single-sample group returns advantage 0.0 (no relative
    comparison is possible).

    Example — rewards [1.0, 0.7, 0.0, 0.0], mean = 0.425::

        normalize=False  →  [+0.575, +0.275, -0.425, -0.425]
        normalize=True   →  (divided by within-group std ≈ 0.406)
                            [+1.416,  +0.677, -1.046, -1.046]
    """
    if len(grouped_rollouts) != len(rewards):
        raise ValueError(
            "grouped_rollouts and rewards must have the same length, got "
            f"{len(grouped_rollouts)} and {len(rewards)}."
        )

    resolved_components = reward_components or [{} for _ in rewards]
    if len(resolved_components) != len(rewards):
        raise ValueError(
            "reward_components and rewards must have the same length, got "
            f"{len(resolved_components)} and {len(rewards)}."
        )

    advantages = _grpo_advantages(list(rewards), normalize=normalize)
    return [
        ScoredGroupedRollout(
            group_id=grb.group_id,
            rollout_index=grb.rollout_index,
            sampling_params=grb.sampling_params,
            final_output=grb.final_output,
            reward=float(reward),
            reward_components=dict(components),
            advantage=float(advantage),
        )
        for grb, reward, components, advantage in zip(
            grouped_rollouts, rewards, resolved_components, advantages
        )
    ]


def format_scored_group_rollout(
    scored_rollouts: "list[ScoredGroupedRollout]",
    *,
    max_obs_chars: int = 200,
) -> str:
    """Print a GRPO prompt group that has already been scored and advantaged.

    Convenience wrapper around :func:`format_group_rollout` for
    :class:`ScoredGroupedRollout` objects — extracts ``rewards`` and
    ``advantages`` automatically so you don't have to re-unpack them::

        scored = assign_group_relative_advantages(grouped, rewards=rewards)
        print(format_scored_group_rollout(scored))
    """
    grouped = [
        GroupedRolloutBatch(
            group_id=s.group_id,
            rollout_index=s.rollout_index,
            sampling_params=s.sampling_params,
            final_output=s.final_output,
        )
        for s in scored_rollouts
    ]
    return format_group_rollout(
        grouped,
        rewards=[s.reward for s in scored_rollouts],
        advantages=[s.advantage for s in scored_rollouts],
        max_obs_chars=max_obs_chars,
    )


def trajectory_log_prob_pack(
    traj: "RolloutTrajectory",
) -> dict[str, list]:
    """Extract the four training-aligned arrays from one rollout trajectory.

    All returned lists have the same length (prompt length + response length)
    so they can be zipped directly in the training loss loop without slicing::

        pack = trajectory_log_prob_pack(traj)
        loss = sum(
            -lp * advantage
            for lp, mask in zip(pack["old_log_probs"], pack["response_mask"])
            if mask
        )

    Mask semantics — ``response_mask`` is 1 only for model-generated action
    tokens inside ``<search>…</search>`` and ``<answer>…</answer>`` spans.
    It is 0 for:

        - prompt tokens  (model didn't generate these)
        - ``<information>…</information>`` observation tokens  (environment,
          not model action — never included in the policy gradient loss)

    If ``old_log_probs`` is ``None`` (backend doesn't implement
    ``compute_log_prob``), this falls back to zeros of the correct length
    so the structure is always valid.

    Returns a dict with keys: ``tokens``, ``attention_mask``,
    ``response_mask``, ``old_log_probs``.
    """
    n = len(traj.tokens or [])
    return {
        "tokens": list(traj.tokens or []),
        "attention_mask": list(traj.attention_mask or [1] * n),
        "response_mask": list(traj.response_mask or [0] * n),
        "old_log_probs": list(traj.old_log_probs or [0.0] * n),
    }


def format_group_rollout(
    grouped_rollouts: "list[GroupedRolloutBatch]",
    *,
    rewards: "list[float] | None" = None,
    advantages: "list[float] | None" = None,
    max_obs_chars: int = 200,
) -> str:
    """Print the strategy diversity across rollouts in one GRPO prompt group.

    Shows what each rollout did — which queries it issued, what it found, and
    what it answered — alongside the reward and group-normalized advantage.
    This is the primary tool for verifying that the group has enough diversity
    for GRPO to compute a meaningful advantage signal::

        grouped = manager.run_prompt_rollout_group(prompt_batch, ...)
        scored  = score_group_rollout(grouped, ground_truth=gt, judge_fn=f)
        print(format_group_rollout(
            grouped,
            rewards=[s.reward for s in scored],
            advantages=[s.advantage for s in scored],
        ))

    Example output::

        Question: Who won the 2024 Nobel Prize in Physics?
        Group: prompt-group-abc  (4 rollouts)
        ────────────────────────────────────────────────────────────
        Rollout 0  temperature=0.80  seed=0
          Step 1: search 2024 Nobel Physics winner
            Observation: Hopfield and Hinton won ...
          Step 2: answer John Hopfield and Geoffrey Hinton.
          [Reward: 1.000 | Advantage: +1.000]
        ────────────────────────────────────────────────────────────
        Rollout 1  temperature=0.95  seed=1
          Step 1: answer directly wrong answer.
          [Reward: 0.000 | Advantage: -1.000]
        ────────────────────────────────────────────────────────────
        Group mean reward: 0.500
    """
    if not grouped_rollouts:
        return ""

    group_id = grouped_rollouts[0].group_id
    question = ""
    logs0 = grouped_rollouts[0].final_output.trajectory_logs
    if logs0:
        question = logs0[0].question

    divider = "─" * 60
    lines: list[str] = [
        f"Question: {question}",
        f"Group: {group_id}  ({len(grouped_rollouts)} rollouts)",
    ]

    for i, grb in enumerate(grouped_rollouts):
        lines.append(divider)
        parts = [f"Rollout {grb.rollout_index}"]
        temp = grb.sampling_params.get("temperature")
        seed = grb.sampling_params.get("seed")
        if temp is not None:
            parts.append(f"temperature={float(temp):.2f}")
        if seed is not None:
            parts.append(f"seed={seed}")
        lines.append("  ".join(parts))

        traj_logs = grb.final_output.trajectory_logs
        if traj_logs:
            reward_i = rewards[i] if rewards is not None and i < len(rewards) else None
            rendered = format_search_trajectory_log(
                traj_logs[0], max_obs_chars=max_obs_chars
            )
            for line in rendered.splitlines():
                lines.append(f"  {line}")
            suffix: list[str] = []
            if reward_i is not None:
                suffix.append(f"Reward: {reward_i:.3f}")
            adv_i = (
                advantages[i]
                if advantages is not None and i < len(advantages)
                else None
            )
            if adv_i is not None:
                sign = "+" if adv_i >= 0 else ""
                suffix.append(f"Advantage: {sign}{adv_i:.3f}")
            if suffix:
                lines.append(f"  [{' | '.join(suffix)}]")

    lines.append(divider)
    if rewards:
        mean_r = sum(rewards) / len(rewards)
        lines.append(f"Group mean reward: {mean_r:.3f}")
    return "\n".join(lines)


def _format_documents(documents: list[RetrievedDocument]) -> str:
    if not documents:
        return _NO_INFO
    parts = []
    for i, doc in enumerate(documents, 1):
        line = f"Doc {i}(Title: {doc.title}) {doc.snippet}".strip()
        if doc.url:
            line += f" URL: {doc.url}"
        parts.append(line)
    return "\n".join(parts)


def _derive_fetch_url(retrieval_url: str) -> str:
    if retrieval_url.endswith("/retrieve"):
        return retrieval_url[: -len("/retrieve")] + "/fetch"
    return retrieval_url.rstrip("/") + "/fetch"


@runtime_checkable
class Retriever(Protocol):
    """Uniform interface for any retrieval backend.

    Implementations: EndpointRetriever (BM25 / dense / hybrid / RAG),
    GoogleRetriever (SerpAPI), SimulateRetriever (LLM-simulated training).
    Adding Tavily, Bing, or any other backend only requires implementing
    this one method — no changes to the dispatch logic.
    """

    def retrieve(self, query: str, topk: int) -> list[RetrievedDocument]:
        """Return structured retrieved documents for one query."""
        ...

    def search(self, query: str, topk: int) -> str:
        """Return retrieved documents formatted for prompt injection."""
        ...


@runtime_checkable
class BatchRetriever(Protocol):
    """Retrieval backend that can serve multiple queries in a single call.

    Backends that implement this avoid N separate HTTP round-trips when the
    agent loop runs several trajectories in parallel.  Check with
    ``isinstance(retriever, BatchRetriever)`` before using.
    """

    def retrieve_batch(
        self, queries: list[str], topk: int
    ) -> list[list[RetrievedDocument]]:
        """Return per-query document lists for all queries in one request."""
        ...


@runtime_checkable
class Fetcher(Protocol):
    """Uniform interface for page-fetch backends used after a model emits <fetch>."""

    def fetch(self, urls: list[str]) -> str:
        """Return fetched page content formatted for the next model turn."""
        ...


@runtime_checkable
class LogProbCapable(Protocol):
    """Backend that can score token log probabilities for PPO/GRPO."""

    def compute_log_prob(self, batch: SearchBatch) -> torch.Tensor | SearchBatch | dict:
        """Return token log probabilities aligned with the response sequence."""
        ...


class EndpointRetriever:
    """POST /retrieve against a local BM25 / dense / hybrid / RAG endpoint."""

    def __init__(self, url: str, retry_attempts: int, sleep_seconds: float) -> None:
        self._url = url
        self._retry_attempts = retry_attempts
        self._sleep_seconds = sleep_seconds

    def retrieve(self, query: str, topk: int) -> list[RetrievedDocument]:
        if not self._url:
            return []
        import requests

        for _ in range(self._retry_attempts):
            try:
                resp = requests.post(
                    self._url, json={"queries": [query], "topk": topk}, timeout=10
                )
                resp.raise_for_status()
                return _documents_from_retrieve_payload(resp.json())
            except Exception:  # pragma: no cover - network dependent
                time.sleep(self._sleep_seconds)
        return []

    def retrieve_batch(
        self, queries: list[str], topk: int
    ) -> list[list[RetrievedDocument]]:
        """Send all queries in a single POST /retrieve and return per-query docs.

        Reduces N individual HTTP calls to one, which matters when many
        trajectories search simultaneously in a training batch.
        """
        if not self._url or not queries:
            return [[] for _ in queries]
        import requests

        for _ in range(self._retry_attempts):
            try:
                resp = requests.post(
                    self._url,
                    json={"queries": queries, "topk": topk},
                    timeout=10,
                )
                resp.raise_for_status()
                return _documents_per_query_from_payload(resp.json(), len(queries))
            except Exception:  # pragma: no cover - network dependent
                time.sleep(self._sleep_seconds)
        return [[] for _ in queries]

    def search(self, query: str, topk: int) -> str:
        return _format_documents(self.retrieve(query, topk))


class EndpointFetcher:
    """POST /fetch against a compatible search backend when available."""

    def __init__(
        self, fetch_url: str, retry_attempts: int, sleep_seconds: float
    ) -> None:
        self._fetch_url = fetch_url
        self._retry_attempts = retry_attempts
        self._sleep_seconds = sleep_seconds

    def fetch(self, urls: list[str]) -> str:
        clean_urls = [url.strip() for url in urls if url.strip()]
        if not clean_urls or not self._fetch_url:
            return _NO_INFO
        import requests

        for _ in range(self._retry_attempts):
            try:
                resp = requests.post(
                    self._fetch_url, json={"urls": clean_urls}, timeout=10
                )
                resp.raise_for_status()
                rows = resp.json().get("result", [])
                return _format_documents([_parse_api_item(item) for item in rows])
            except Exception:  # pragma: no cover - network dependent
                time.sleep(self._sleep_seconds)
        return _NO_INFO


class GoogleRetriever:
    """SerpAPI Google search backend."""

    def __init__(self, api_key: str, retry_attempts: int, sleep_seconds: float) -> None:
        self._api_key = api_key
        self._retry_attempts = retry_attempts
        self._sleep_seconds = sleep_seconds

    def retrieve(self, query: str, topk: int) -> list[RetrievedDocument]:
        if not self._api_key:
            return []
        import serpapi  # optional dep

        params = {"engine": "google", "q": query, "api_key": self._api_key, "num": topk}
        for attempt in range(self._retry_attempts):
            try:
                items = serpapi.search(params).get("organic_results", [])
                return [_parse_api_item(item) for item in items]
            except Exception:  # pragma: no cover - network dependent
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._sleep_seconds)
        return []

    def search(self, query: str, topk: int) -> str:
        return _format_documents(self.retrieve(query, topk))


class SimulateRetriever:
    """LLM-simulated search for training (simulate_sft / simulate_prompt).

    Not a real retrieval backend — satisfies Retriever so training code
    handles all modes uniformly without an if/elif on search_mode.
    """

    def __init__(
        self,
        llm_ip: str | None,
        temperature: float,
        llm_max_retries: int,
        *,
        problem: str,
        ground_truth: str,
        gt_threshold: float,
        use_examples: bool = False,
    ) -> None:
        self._llm_ip = llm_ip
        self._temperature = temperature
        self._llm_max_retries = llm_max_retries
        self._problem = problem
        self._ground_truth = ground_truth
        self._gt_threshold = gt_threshold
        self._use_examples = use_examples

    def search(self, query: str, topk: int) -> str:
        return (
            search_simulate(
                self._llm_ip,
                topk,
                self._temperature,
                query,
                self._problem,
                self._ground_truth,
                self._gt_threshold,
                use_examples=self._use_examples,
                llm_max_retries=self._llm_max_retries,
            )
            or _NO_INFO
        )

    def retrieve(self, query: str, topk: int) -> list[RetrievedDocument]:
        text = self.search(query, topk)
        if not text or text == _NO_INFO:
            return []
        documents: list[RetrievedDocument] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            title = "Simulated retrieval result"
            snippet = line
            if ":" in line:
                prefix, remainder = line.split(":", 1)
                title = prefix.strip() or title
                snippet = remainder.strip() or line
            documents.append(RetrievedDocument(title=title, snippet=snippet))
        return documents


@dataclass(frozen=True)
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    llm_ip: str | None = None
    retriever_ip: str | None = None
    retrieval_url: str | None = None  # full URL for the local retrieval_server endpoint
    temperature: float = 0.8
    topk: int = 5
    search_mode: str = "google"
    end_threshold: float = 0.5
    start_threshold: float = 0.5
    llm_max_retries: int = 5
    search_max_workers: int = 10
    wiki_retry_attempts: int = 10
    google_retry_attempts: int = 3
    wiki_retry_sleep_seconds: float = 1.0
    google_retry_sleep_seconds: float = 2.0


def _normalize_ip_list(ip_list_raw: str | None) -> list[str]:
    if not ip_list_raw:
        return []
    return [ip.strip() for ip in ip_list_raw.split(",") if ip.strip()]


def _resolve_ground_truth_text(ground_truth: Any) -> str:
    if isinstance(ground_truth, (list, tuple)):
        return str(ground_truth[0]) if ground_truth else ""
    return str(ground_truth)


def ask_llm(
    ip_list_raw: str | None, prompt: str, temperature: float, max_retries: int = 5
) -> str:
    """Call one of the configured LLM endpoints with bounded retries."""

    ip_list = _normalize_ip_list(ip_list_raw)
    if not ip_list:
        raise ValueError("At least one llm_ip must be provided.")

    last_error: Exception | None = None
    for _ in range(max_retries):
        ip = random.choice(ip_list)
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://{ip}:6001/v1",
            )
            response = client.chat.completions.create(
                model="",
                max_tokens=600,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            time.sleep(0.2)

    raise RuntimeError("LLM request failed after retries.") from last_error


def search_simulate(
    ip: str | None,
    topk: int,
    temperature: float,
    query: str,
    problem: str,
    ground_truth: str,
    gt_threshold: float,
    use_examples: bool = False,
    llm_max_retries: int = 5,
) -> str:
    useful = random.random() > gt_threshold
    if use_examples:
        if useful:
            prompt = f"""You are the Google search engine.
Given a query, you need to imitate the style of the following demos and generate five useful documents for the query.

Here is an example:
Query: George Washington Bridge opening year
Useful Output:
Doc 1: The George Washington Bridge, an iconic structure connecting New York City to New Jersey, opened on October 25, 1931. Designed by Othmar Ammann, it marked a major milestone in civil engineering.
Doc 2: Originally the Hudson River Bridge, the George Washington Bridge was named after the U.S.'s first president. Its 3,500-foot suspension span was the world's longest at completion in 1931.
Doc 3: The bridge was modified in 1962 with added lower deck lanes, increasing capacity and easing congestion. This expansion transformed the bridge into a double-decked structure with twelve lanes.
Doc 4: Constructed over four years, the George Washington Bridge's steel towers and cables exemplified engineering progress. It crucially linked New York and New Jersey's transportation networks.
Doc 5: Handling over 103 million annual vehicles, the George Washington Bridge is globally one of the busiest. The Port Authority of NY and NJ oversees its traffic and infrastructure maintenance.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
You should generate documents that can help the user find the answer.
Each document should contain about 30 words.
You must directly output the English documents and not output any other texts.

Query: {query}
Useful Output:
"""
        else:
            prompt = f"""You are the Google search engine.
Given a query, you need to imitate the style of the following demos and generate five related but noisy documents for the query.

Here is an example:
Query: George Washington Bridge opening year
Noisy Output:
Doc 1: The George Washington Bridge was a significant addition to New York's infrastructure, but there were many challenges during its construction, including budget overruns and worker strikes.
Doc 2: While often mistaken for the opening of another iconic bridge, the Brooklyn Bridge, the George Washington Bridge has its own storied history involving political maneuvering and urban planning debates that shaped the bridge's final design.
Doc 3: Discussions about the naming of the George Washington Bridge are interesting, as it was named to honor the first President of the United States. The name choice was significant given the geographical and historical implications of Washington's legacy.
Doc 4: The bridge has been a point of contention over toll increases, traffic congestion solutions, and environmental impact assessments, which continue to affect policy decisions in the area.
Doc 5: Considerations around the visual and architectural design of the George Washington Bridge also played a crucial role in its legacy as a landmark, balancing aesthetics with functionality.

Each document should contain about 30 words, and these documents should contain related but noisy information.
You must directly output the English documents and not output any other texts.

Query: {query}
Noisy Output:
"""
    else:
        if useful:
            prompt = f"""You are the Google search engine.
Given a query, you need to generate five useful documents for the query.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
Each document should contain about 30 words, and these documents should contain useful information.

Query: {query}
Useful Output:
"""
        else:
            prompt = f"""You are the Google search engine.
Given a query, you need to generate five noisy documents for the query.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
Each document should contain about 30 words, and these documents should contain noisy information.

Query: {query}
Noisy Output:
"""

    results = ask_llm(ip, prompt, temperature, max_retries=llm_max_retries)
    return "\n".join(results.replace("\n\n", "\n").split("\n")).split(
        f"Doc {topk + 1}"
    )[0]


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer: Any,
        config: GenerationConfig,
        is_validation: bool = False,
        generation_backend: Any | None = None,
        actor_rollout_wg: Any | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.generation_backend = generation_backend or actor_rollout_wg
        if self.generation_backend is None:
            raise ValueError("generation_backend is required.")
        self.config = config
        self.is_validation = is_validation
        self.fetcher = fetcher
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

    def _batch_tokenize(self, responses: list[str]) -> torch.Tensor:
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def _postprocess_responses(
        self, responses: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        responses_str = [
            self._truncate_to_first_action(response) for response in responses_str
        ]

        return self._batch_tokenize(responses_str), responses_str

    def _truncate_to_first_action(self, response: str) -> str:
        """Keep only the first complete action block emitted by the policy."""

        earliest_match = None
        earliest_start = None
        for tag in ROLLOUT_ACTION_TAGS:
            end_tag = f"</{tag}>"
            end_index = response.find(end_tag)
            if end_index == -1:
                continue
            start_tag = f"<{tag}>"
            start_index = response.rfind(start_tag, 0, end_index + len(end_tag))
            if start_index == -1:
                continue
            if earliest_start is None or start_index < earliest_start:
                earliest_start = start_index
                earliest_match = response[start_index : end_index + len(end_tag)]
        return earliest_match if earliest_match is not None else response

    @staticmethod
    def _safe_truncate_observation(obs: str, max_chars: int) -> str:
        """Truncate a long observation while preserving structural closing tags.

        A hard character cut can leave the context with a dangling
        ``<information>`` block that never closes, confusing the model on the
        next generation step.  This method truncates the *content* instead,
        always re-appending the closing tag so the context stays well-formed.
        """
        if len(obs) <= max_chars:
            return obs
        close = "</information>\n\n"
        if close in obs:
            before_close = obs[: obs.rindex(close)]
            budget = max_chars - len(close) - 3  # 3 for "..."
            if budget > 0:
                return before_close[:budget] + "..." + close
        return obs[:max_chars]

    def _process_next_obs(self, next_obs: list[str]) -> torch.Tensor:
        # Pre-truncate at character level so the token cut never lands inside
        # a closing tag.  4 chars ≈ 1 token is conservative for most tokenizers.
        max_chars = self.config.max_obs_length * 4
        next_obs = [self._safe_truncate_observation(obs, max_chars) for obs in next_obs]
        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]
        return next_obs_ids

    def build_react_context_transitions(
        self,
        responses_str: list[str],
        next_obs: list[str],
    ) -> list[ReActContextTransition]:
        """Build ReAct context transitions for the next generation step.

        Each transition explicitly captures the text appended back into context
        after one action:

            context += model_output
            context += retrieval_result / tool observation
        """
        if len(responses_str) != len(next_obs):
            raise ValueError("responses_str and next_obs must have the same length.")
        return [
            ReActContextTransition(
                action_text=action_text,
                observation_text=observation_text,
                appended_context_text=action_text + observation_text,
            )
            for action_text, observation_text in zip(responses_str, next_obs)
        ]

    def _update_rolling_state(
        self,
        rollings: SearchBatch,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor,
    ) -> SearchBatch:
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        effective_len = int(new_attention_mask.sum(dim=1).max().item())
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = SearchBatch.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(getattr(rollings, "meta_info", {}))
        return new_rollings

    def apply_react_context_transitions(
        self,
        rollings: SearchBatch,
        cur_responses: torch.Tensor,
        responses_str: list[str],
        next_obs: list[str],
    ) -> tuple[SearchBatch, torch.Tensor, list[ReActContextTransition]]:
        """Append action outputs and observations into the next-step context."""
        transitions = self.build_react_context_transitions(
            responses_str,
            next_obs,
        )
        next_obs_ids = self._process_next_obs(next_obs)
        updated_rollings = self._update_rolling_state(
            rollings,
            cur_responses,
            next_obs_ids,
        )
        return updated_rollings, next_obs_ids, transitions

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor | None = None,
        pad_to_left: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(
                info.size(), pad_id, dtype=info.dtype, device=info.device
            )
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        padded_tensor = self.tensor_fn.concatenate_with_padding(
            [concatenated],
            pad_to_left=pad_to_left,
        )
        padded_tensor_with_info = self.tensor_fn.concatenate_with_padding(
            [concatenated_with_info],
            pad_to_left=pad_to_left,
        )
        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self,
        right_side: dict[str, torch.Tensor],
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if next_obs_ids is not None:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False,
                )
            )
        else:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    pad_to_left=False,
                )
            )

        effective_len = int(
            self.tensor_fn.create_attention_mask(responses).sum(dim=1).max().item()
        )
        max_len = min(self.config.max_prompt_length, effective_len)
        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: SearchBatch) -> SearchBatch:
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.generation_backend.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch:
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.generation_backend.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for key, value in active_batch.batch.items():
            pad_sequence = value[0:1].repeat(padding_size, *[1] * (value.dim() - 1))
            padded_batch[key] = torch.cat([value, pad_sequence], dim=0)

        padded_active_batch = SearchBatch.from_dict(padded_batch)
        for key in padded_active_batch.batch:
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_output = self.generation_backend.generate_sequences(padded_active_batch)
        trimmed_batch = {
            key: value[:-padding_size] for key, value in padded_output.batch.items()
        }
        trimmed_meta = {}
        for key, value in getattr(padded_output, "meta_info", {}).items():
            trimmed_meta[key] = (
                value[:-padding_size] if isinstance(value, torch.Tensor) else value
            )

        padded_output.batch = trimmed_batch
        padded_output.meta_info = trimmed_meta
        return padded_output

    def _response_action_mask(self, batch: SearchBatch) -> torch.Tensor:
        """Mask for model-generated action tokens inside the response sequence."""

        return self.tensor_fn.create_attention_mask(
            batch.batch["responses_with_info_mask"]
        ).to(dtype=torch.float32)

    def _normalize_log_prob_output(
        self,
        result: torch.Tensor | SearchBatch | dict[str, Any],
    ) -> torch.Tensor:
        if isinstance(result, torch.Tensor):
            return result
        if isinstance(result, SearchBatch):
            if "log_probs" in result.batch:
                return result.batch["log_probs"]
            if "log_probs" in result.meta_info:
                return result.meta_info["log_probs"]
        if isinstance(result, dict) and "log_probs" in result:
            return result["log_probs"]
        raise TypeError(
            "compute_log_prob(...) must return a Tensor, SearchBatch with `log_probs`, "
            "or dict containing `log_probs`."
        )

    def compute_log_prob(
        self,
        batch: SearchBatch,
        *,
        backend: LogProbCapable | None = None,
        store_key: str = "new_log_probs",
        overwrite: bool = True,
    ) -> torch.Tensor:
        """Recompute token log probabilities aligned with the rollout response.

        Used twice in PPO/GRPO-style training:
        - `old_log_probs`: under the policy that generated the rollout
        - `new_log_probs`: under the current policy being optimized

        When `store_key='old_log_probs'` and `overwrite=False`, returns the
        already-stored tensor without recomputing.  This protects the frozen
        rollout log probs from being silently overwritten during PPO updates.
        """
        if not overwrite and store_key in batch.batch:
            return batch.batch[store_key]

        scorer = backend or self.generation_backend
        if not isinstance(scorer, LogProbCapable):
            raise TypeError(
                "generation backend does not implement compute_log_prob(...)"
            )

        raw = scorer.compute_log_prob(batch)
        log_probs = self._normalize_log_prob_output(raw).to(dtype=torch.float32)
        if log_probs.shape != batch.batch["responses"].shape:
            raise ValueError(
                "compute_log_prob(...) must return log_probs aligned with `responses` "
                f"shape {tuple(batch.batch['responses'].shape)}, got {tuple(log_probs.shape)}."
            )

        masked_log_probs = log_probs * self._response_action_mask(batch)
        batch.batch[store_key] = masked_log_probs
        self._attach_log_probs_to_trajectories(batch, store_key)
        return masked_log_probs

    def _attach_log_probs_to_trajectories(
        self,
        batch: SearchBatch,
        store_key: str,
    ) -> None:
        if store_key not in {"old_log_probs", "new_log_probs"}:
            raise ValueError(
                f"store_key must be 'old_log_probs' or 'new_log_probs', got {store_key!r}"
            )
        trajectories = batch.non_tensor_batch.get("trajectories", [])
        log_probs = batch.batch.get(store_key)
        if log_probs is None or not trajectories:
            return
        batch.non_tensor_batch["trajectories"] = [
            dataclass_replace(
                traj,
                **{
                    store_key: (
                        # Prepend prompt-length zeros so old/new_log_probs aligns with
                        # traj.tokens (prompt + response) and traj.response_mask.
                        # Training loops can then zip all four arrays without slicing:
                        #   tokens, attention_mask, response_mask, old_log_probs
                        [0.0] * len(traj.prompt_token_ids)
                        + (log_probs[i].tolist() if i < log_probs.shape[0] else [])
                    )
                },
            )
            for i, traj in enumerate(trajectories)
        ]

    def prepare_policy_log_probs(
        self,
        batch: SearchBatch,
        *,
        old_backend: LogProbCapable | None = None,
        new_backend: LogProbCapable | None = None,
        compute_ratio: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare old/new token log probabilities for PPO / GRPO updates.

        This is the standard policy-evaluation step needed before computing
        PPO/GRPO ratios:

            old_log_probs = log pi_theta_old(a_t | s_t)
            new_log_probs = log pi_theta(a_t | s_t)

        The tensors are stored on ``batch.batch`` under
        ``"old_log_probs"`` and ``"new_log_probs"`` and mirrored into each
        ``RolloutTrajectory`` when present. When ``compute_ratio=True`` this
        also computes ``prob_ratio = exp(new - old)``.
        """
        resolved_old_backend = old_backend or self.generation_backend
        resolved_new_backend = new_backend or self.generation_backend

        old_log_probs = self.compute_log_prob(
            batch,
            backend=resolved_old_backend,
            store_key="old_log_probs",
        )
        new_log_probs = self.compute_log_prob(
            batch,
            backend=resolved_new_backend,
            store_key="new_log_probs",
        )
        if compute_ratio:
            self.compute_prob_ratio(batch)
        batch.meta_info["policy_log_probs_prepared"] = True
        return old_log_probs, new_log_probs

    def compute_prob_ratio(self, batch: SearchBatch) -> torch.Tensor:
        """Compute the per-token PPO / GRPO probability ratio r_t(θ).

        r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
               = exp(new_log_probs − old_log_probs)

        Requires both compute_log_prob(..., store_key='old_log_probs') and
        compute_log_prob(..., store_key='new_log_probs') to have been called.
        Env-injected observation tokens get ratio 1.0 (no gradient contribution).
        Result is stored in batch.batch['prob_ratio'] and returned.
        """
        old_lp = batch.batch.get("old_log_probs")
        new_lp = batch.batch.get("new_log_probs")
        if old_lp is None:
            raise ValueError(
                "old_log_probs not found; call compute_log_prob(..., store_key='old_log_probs') first."
            )
        if new_lp is None:
            raise ValueError(
                "new_log_probs not found; call compute_log_prob(..., store_key='new_log_probs') first."
            )
        action_mask = self._response_action_mask(batch)
        log_ratio = torch.clamp(
            (new_lp - old_lp) * action_mask,
            min=-_LOG_RATIO_CLAMP,
            max=_LOG_RATIO_CLAMP,
        )
        ratio = torch.exp(log_ratio)
        batch.batch["prob_ratio"] = ratio
        return ratio

    def compute_action_type_masks(self, batch: SearchBatch) -> dict[str, torch.Tensor]:
        """Binary masks over response token positions, one per action type.

        Returns a dict mapping each action tag — ``"search"``, ``"plan"``,
        ``"fetch"``, ``"answer"`` — to a float32 tensor of shape
        ``[batch_size, response_length]``.  Position ``[i, t]`` is ``1.0`` iff
        token ``t`` in rollout ``i`` was emitted inside a ``<tag>…</tag>`` block
        of that type.

        Observation tokens injected by the environment are never part of a
        policy action block and always get ``0.0`` across all masks.
        """
        response_ids_list = batch.batch["responses"].tolist()
        B, T = batch.batch["responses"].shape
        masks = {
            tag: torch.zeros(B, T, dtype=torch.float32) for tag in ROLLOUT_ACTION_TAGS
        }
        if not hasattr(self.tokenizer, "decode"):
            return masks
        for row, ids in enumerate(response_ids_list):
            offsets = _token_char_offsets(ids, self.tokenizer)
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            for m in ACTION_PATTERN.finditer(text):
                tag = m.group("tag")
                span_start, span_end = m.start(), m.end()
                for tok, (tok_start, tok_end) in enumerate(offsets):
                    if (
                        tok_start >= span_start
                        and tok_end <= span_end
                        and tok_start < tok_end
                    ):
                        masks[tag][row, tok] = 1.0
        return masks

    def compute_policy_family_masks(
        self,
        batch: SearchBatch,
    ) -> dict[str, torch.Tensor]:
        """Higher-level masks for search / reasoning / stopping / answer policies.

        The rollout emits explicit action tags, but PPO / GRPO discussions often
        refer to broader policy families. In this codebase the mapping is:

        - ``search_policy``    -> ``<search> ... </search>``
        - ``reasoning_policy`` -> ``<plan> ... </plan>`` and ``<fetch> ... </fetch>``
        - ``stopping_policy``  -> terminal ``<answer> ... </answer>``
        - ``answer_policy``    -> ``<answer> ... </answer>``

        ``stopping_policy`` and ``answer_policy`` intentionally share the same
        token span because the stop decision is represented by emitting the
        final answer action.
        """
        type_masks = self.compute_action_type_masks(batch)
        answer_mask = type_masks["answer"]
        reasoning_mask = torch.clamp(
            type_masks["plan"] + type_masks["fetch"],
            min=0.0,
            max=1.0,
        )
        return {
            "search_policy": type_masks["search"],
            "reasoning_policy": reasoning_mask,
            "stopping_policy": answer_mask,
            "answer_policy": answer_mask,
        }

    def policy_loss_breakdown(
        self,
        batch: SearchBatch,
        *,
        advantages: torch.Tensor | None = None,
        config: PPOPolicyLossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-action-type clipped policy objective for diagnostic logging.

        Computes the mean clipped surrogate contribution from each action type
        (search, plan, fetch, answer) independently.  The four values together
        describe where the gradient signal is coming from and whether one action
        type dominates training.

        Does not modify ``batch``; call ``compute_prob_ratio`` first if
        ``prob_ratio`` is not already in ``batch.batch``.
        """
        cfg = config or PPOPolicyLossConfig()
        ratio = batch.batch.get("prob_ratio")
        if ratio is None:
            ratio = self.compute_prob_ratio(batch)
        resolved_advantages = advantages
        if resolved_advantages is None:
            resolved_advantages = batch.batch.get("advantages")
        if resolved_advantages is None:
            raise ValueError(
                "advantages not found; pass `advantages=` or populate batch.batch['advantages'] first."
            )
        resolved_advantages = resolved_advantages.to(dtype=torch.float32)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - float(cfg.clip_epsilon),
            1.0 + float(cfg.clip_epsilon),
        )
        surrogate = torch.minimum(
            ratio * resolved_advantages, clipped_ratio * resolved_advantages
        )
        type_masks = self.compute_action_type_masks(batch)
        breakdown: dict[str, torch.Tensor] = {}
        for tag, mask in type_masks.items():
            n_tokens = torch.clamp(mask.sum(), min=1.0)
            breakdown[tag] = (surrogate * mask).sum() / n_tokens
        return breakdown

    def policy_update_breakdown(
        self,
        batch: SearchBatch,
        *,
        advantages: torch.Tensor | None = None,
        config: PPOPolicyLossConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-policy-family clipped objective for PPO / GRPO diagnostics."""
        cfg = config or PPOPolicyLossConfig()
        ratio = batch.batch.get("prob_ratio")
        if ratio is None:
            ratio = self.compute_prob_ratio(batch)
        resolved_advantages = advantages
        if resolved_advantages is None:
            resolved_advantages = batch.batch.get("advantages")
        if resolved_advantages is None:
            raise ValueError(
                "advantages not found; pass `advantages=` or populate batch.batch['advantages'] first."
            )
        resolved_advantages = resolved_advantages.to(dtype=torch.float32)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - float(cfg.clip_epsilon),
            1.0 + float(cfg.clip_epsilon),
        )
        surrogate = torch.minimum(
            ratio * resolved_advantages, clipped_ratio * resolved_advantages
        )
        family_masks = self.compute_policy_family_masks(batch)
        breakdown: dict[str, torch.Tensor] = {}
        for family, mask in family_masks.items():
            n_tokens = torch.clamp(mask.sum(), min=1.0)
            breakdown[family] = (surrogate * mask).sum() / n_tokens
        return breakdown

    def compute_policy_loss(
        self,
        batch: SearchBatch,
        *,
        advantages: torch.Tensor | None = None,
        config: PPOPolicyLossConfig | None = None,
    ) -> torch.Tensor:
        """Compute the clipped PPO / GRPO policy loss over action tokens.

        This implements the standard clipped surrogate objective:

            L_clip(theta) =
                E[min(r_t(theta) A_t,
                      clip(r_t(theta), 1-eps, 1+eps) A_t)]

        The returned tensor is the negated objective so it can be minimized by
        gradient descent. The same token-level policy is used for search,
        reasoning, stopping, and answer generation, so one loss updates all of
        them wherever `responses_with_info_mask` marks model-chosen action
        tokens.
        """
        cfg = config or PPOPolicyLossConfig()
        action_mask = self._response_action_mask(batch)
        ratio = batch.batch.get("prob_ratio")
        if ratio is None:
            ratio = self.compute_prob_ratio(batch)

        resolved_advantages = advantages
        if resolved_advantages is None:
            resolved_advantages = batch.batch.get("advantages")
        if resolved_advantages is None:
            raise ValueError(
                "advantages not found; pass `advantages=` or populate batch.batch['advantages'] first."
            )
        resolved_advantages = resolved_advantages.to(dtype=torch.float32)
        if resolved_advantages.shape != ratio.shape:
            raise ValueError(
                "advantages must align with prob_ratio shape "
                f"{tuple(ratio.shape)}, got {tuple(resolved_advantages.shape)}."
            )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - float(cfg.clip_epsilon),
            1.0 + float(cfg.clip_epsilon),
        )
        surrogate = (
            torch.minimum(
                ratio * resolved_advantages, clipped_ratio * resolved_advantages
            )
            * action_mask
        )

        # Compute action-type masks once — shared by action_type_weights reweighting
        # and the policy-family breakdown below.  Without this, compute_action_type_masks
        # would be called 2–3 times (one tokenizer decode pass per call).
        type_masks = self.compute_action_type_masks(batch)

        if cfg.action_type_weights:
            weight_tensor = torch.ones_like(ratio)
            for tag, w in cfg.action_type_weights.items():
                if tag in type_masks:
                    weight_tensor = weight_tensor + (w - 1.0) * type_masks[tag].to(
                        ratio.device
                    )
            surrogate = surrogate * weight_tensor

        normalizer = torch.clamp(action_mask.sum(), min=1.0)
        policy_objective = surrogate.sum() / normalizer

        kl_term = torch.tensor(0.0, dtype=torch.float32, device=policy_objective.device)
        if cfg.kl_coefficient:
            kl = batch.batch.get("per_token_kl")
            if kl is None:
                kl = self.per_token_kl(batch)
            kl_term = (kl * action_mask).sum() / normalizer

        # Entropy bonus: H(π) ≈ −E[log π(a|s)] over action tokens.
        # Adding this to the objective keeps the search / reasoning / stopping
        # policies diverse and prevents premature mode collapse.
        entropy_term = torch.tensor(
            0.0, dtype=torch.float32, device=policy_objective.device
        )
        if cfg.entropy_coefficient:
            new_lp = batch.batch.get("new_log_probs")
            if new_lp is not None:
                entropy_term = -(new_lp * action_mask).sum() / normalizer

        loss = (
            -policy_objective
            + float(cfg.kl_coefficient) * kl_term
            - float(cfg.entropy_coefficient) * entropy_term
        )
        clip_fraction = (
            (ratio != clipped_ratio).to(dtype=torch.float32) * action_mask
        ).sum() / normalizer

        # Policy-family breakdown from the already-computed type_masks — no extra
        # decode pass.  stopping_policy and answer_policy share the <answer> span.
        answer_mask = type_masks["answer"]
        reasoning_mask = torch.clamp(
            type_masks["plan"] + type_masks["fetch"], min=0.0, max=1.0
        )
        family_masks = {
            "search_policy": type_masks["search"],
            "reasoning_policy": reasoning_mask,
            "stopping_policy": answer_mask,
            "answer_policy": answer_mask,
        }
        family_breakdown: dict[str, float] = {}
        family_token_counts: dict[str, float] = {}
        for family, fmask in family_masks.items():
            n_fam = torch.clamp(fmask.sum(), min=1.0)
            family_breakdown[family] = float(
                ((surrogate * fmask).sum() / n_fam).detach().item()
            )
            family_token_counts[family] = float(fmask.sum().item())

        batch.batch["advantages"] = resolved_advantages * action_mask
        batch.batch["clipped_prob_ratio"] = clipped_ratio
        batch.batch["policy_loss"] = loss
        batch.batch["policy_objective"] = policy_objective.detach()
        batch.batch["clip_fraction"] = clip_fraction.detach()
        batch.batch["mean_advantage"] = (
            (resolved_advantages * action_mask).sum() / normalizer
        ).detach()
        batch.batch["mean_kl"] = kl_term.detach()
        if cfg.entropy_coefficient:
            batch.batch["entropy"] = entropy_term.detach()
        batch.meta_info["policy_update_token_counts"] = family_token_counts
        batch.meta_info["updated_policies"] = [
            family for family, count in family_token_counts.items() if count > 0.0
        ]
        batch.meta_info["policy_update_breakdown"] = family_breakdown
        return loss

    def per_token_kl(self, batch: SearchBatch) -> torch.Tensor:
        """Compute per-token KL divergence KL(π_θ_old ‖ π_θ) for GRPO regularisation.

        Uses the unbiased estimator:
            KL(p ‖ q) ≈ exp(log_p − log_q) − (log_p − log_q) − 1

        Env-injected observation tokens contribute 0 KL (masked out).
        Result is stored in batch.batch['per_token_kl'] and returned.
        """
        old_lp = batch.batch.get("old_log_probs")
        new_lp = batch.batch.get("new_log_probs")
        if old_lp is None or new_lp is None:
            raise ValueError(
                "Both old_log_probs and new_log_probs must be computed before calling per_token_kl."
            )
        action_mask = self._response_action_mask(batch)
        log_ratio = torch.clamp(
            (old_lp - new_lp) * action_mask,
            min=-_LOG_RATIO_CLAMP,
            max=_LOG_RATIO_CLAMP,
        )
        kl = (torch.exp(log_ratio) - log_ratio - 1.0) * action_mask
        batch.batch["per_token_kl"] = kl
        return kl

    def _initialize_agent_loop_state(
        self,
        gen_batch: SearchBatch,
        initial_input_ids: torch.Tensor,
    ) -> AgentLoopState:
        batch_size = gen_batch.batch["input_ids"].shape[0]
        return AgentLoopState(
            rollings=gen_batch,
            original_left_side={
                "input_ids": initial_input_ids[:, -self.config.max_start_length :]
            },
            original_right_side={
                "responses": initial_input_ids[:, []],
                "responses_with_info_mask": initial_input_ids[:, []],
            },
            active_mask=torch.ones(batch_size, dtype=torch.bool),
            trajectory_turns=[0 for _ in range(batch_size)],
            turns_stats=torch.ones(batch_size, dtype=torch.int),
            valid_action_stats=torch.zeros(batch_size, dtype=torch.int),
            valid_search_stats=torch.zeros(batch_size, dtype=torch.int),
            active_num_list=[batch_size],
        )

    def _trim_rollings_for_generation(self, rollings: SearchBatch) -> SearchBatch:
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=["input_ids", "attention_mask", "position_ids"],
        )
        return rollings

    def _select_active_batch(
        self,
        rollings: SearchBatch,
        active_mask: torch.Tensor,
    ) -> SearchBatch:
        return SearchBatch.from_dict(
            {key: value[active_mask] for key, value in rollings.batch.items()}
        )

    def _run_active_generation_step(
        self,
        gen_batch: SearchBatch,
        rollings: SearchBatch,
        *,
        search_mode: str,
        gt_threshold: float,
        active_mask: torch.Tensor,
        do_search: bool = True,
        search_cache: dict[tuple[int, str], str] | None = None,
    ) -> tuple[AgentLoopStepResult, dict[str, Any]]:
        rollout_step, meta_info = self.actor_rollout_step(
            rollings,
            active_mask=active_mask,
        )
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            rollout_step.responses_str,
            gen_batch.non_tensor_batch["question"],
            gen_batch.non_tensor_batch["golden_answers"],
            search_mode,
            gt_threshold,
            active_mask,
            do_search=do_search,
            search_cache=search_cache,
        )
        return (
            AgentLoopStepResult(
                responses_ids=rollout_step.responses_ids,
                responses_str=rollout_step.responses_str,
                parsed_actions=rollout_step.parsed_actions,
                next_obs=next_obs,
                dones=dones,
                valid_action=valid_action,
                is_search=is_search,
            ),
            meta_info,
        )

    def actor_rollout_step(
        self,
        rollings: SearchBatch,
        *,
        active_mask: torch.Tensor,
    ) -> tuple[ActorRolloutStep, dict[str, Any]]:
        """Sample one agent action step from the current policy.

        This is the actor rollout primitive:

            state(SearchBatch) -> action tokens -> parsed PolicyAction

        The sampled action can be a search, plan, fetch, or final answer. Tool
        execution and observation appending happen afterwards in the outer agent
        loop.
        """
        gen_output = self._generate_with_gpu_padding(
            self._select_active_batch(rollings, active_mask)
        )
        meta_info = getattr(gen_output, "meta_info", {}) or {}
        responses_ids, responses_str = self._postprocess_responses(
            gen_output.batch["responses"]
        )
        responses_ids, responses_str = self.tensor_fn.example_level_pad(
            responses_ids,
            responses_str,
            active_mask,
        )
        parsed_actions = self.parse_policy_actions(responses_str)
        return (
            ActorRolloutStep(
                responses_ids=responses_ids,
                responses_str=responses_str,
                parsed_actions=parsed_actions,
            ),
            meta_info,
        )

    def _apply_step_result(
        self,
        state: AgentLoopState,
        step_result: AgentLoopStepResult,
        *,
        include_observations: bool,
    ) -> None:
        # Snapshot which trajectories participated in THIS turn before updating
        # the mask.  Trajectories that were already done carry dummy
        # responses_str / parsed_actions and must not pollute the step history.
        participating = state.active_mask.tolist()

        curr_active_mask = torch.tensor(
            [not done for done in step_result.dones], dtype=torch.bool
        )
        state.active_mask = state.active_mask & curr_active_mask
        state.active_num_list.append(int(state.active_mask.sum().item()))

        if include_observations:
            state.turns_stats[curr_active_mask] += 1
            state.rollings, next_obs_ids, transitions = (
                self.apply_react_context_transitions(
                    state.rollings,
                    step_result.responses_ids,
                    step_result.responses_str,
                    step_result.next_obs,
                )
            )
            state.original_right_side = self._update_right_side(
                state.original_right_side,
                step_result.responses_ids,
                next_obs_ids,
            )
            turn_num = len(state.meta_info.get("react_trajectory", []))
            state.meta_info.setdefault("react_trajectory", []).append(
                [
                    ReActStep(
                        turn=turn_num,
                        action_tag=action.tag,
                        action_content=action.content,
                        observation=obs,
                        is_terminal=action.tag == "answer",
                    )
                    if was_active
                    else None
                    for action, obs, was_active in zip(
                        step_result.parsed_actions, step_result.next_obs, participating
                    )
                ]
            )
            state.meta_info.setdefault("context_transitions", []).append(
                [
                    transition if was_active else None
                    for transition, was_active in zip(transitions, participating)
                ]
            )
            state.meta_info.setdefault("context_token_lengths", []).append(
                int(state.rollings.batch["attention_mask"].sum(dim=1).max().item())
            )
        else:
            state.original_right_side = self._update_right_side(
                state.original_right_side,
                step_result.responses_ids,
            )

        state.valid_action_stats += torch.tensor(
            step_result.valid_action, dtype=torch.int
        )
        state.valid_search_stats += torch.tensor(step_result.is_search, dtype=torch.int)

    def _decode_context_rows(self, input_ids: torch.Tensor) -> list[str]:
        if hasattr(self.tokenizer, "batch_decode"):
            try:
                return list(
                    self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                )
            except Exception:
                pass
        if hasattr(self.tokenizer, "decode"):
            decoded_rows: list[str] = []
            for row in input_ids:
                try:
                    decoded_rows.append(
                        self.tokenizer.decode(row.tolist(), skip_special_tokens=False)
                    )
                except Exception:
                    decoded_rows.append(" ".join(str(int(tok)) for tok in row.tolist()))
            return decoded_rows
        return [" ".join(str(int(tok)) for tok in row.tolist()) for row in input_ids]

    def _snapshot_rollout_states(self, rollings: SearchBatch) -> list[str]:
        """Decode the current per-trajectory context before the next action."""
        input_ids = rollings.batch.get("input_ids")
        if input_ids is None:
            return []
        return self._decode_context_rows(input_ids)

    def _record_search_steps(
        self,
        state: AgentLoopState,
        step_result: AgentLoopStepResult,
        state_snapshots: list[str],
    ) -> None:
        """Record step-by-step RL trajectory logs for every participating prompt."""
        step_history = state.meta_info.setdefault("search_step_history", [])
        participating = state.active_mask.tolist()
        current_turn = len(step_history) + 1
        per_turn_steps: list[SearchStep | None] = []
        for batch_index, (
            was_active,
            action,
            model_output,
            observation,
            done,
        ) in enumerate(
            zip(
                participating,
                step_result.parsed_actions,
                step_result.responses_str,
                step_result.next_obs,
                step_result.dones,
            )
        ):
            if not was_active:
                per_turn_steps.append(None)
                continue
            action_type = action.tag if action.tag is not None else "invalid"
            per_turn_steps.append(
                SearchStep(
                    step_id=current_turn,
                    state=state_snapshots[batch_index]
                    if batch_index < len(state_snapshots)
                    else "",
                    model_output=model_output,
                    action_type=action_type,
                    action_value=action.content,
                    observation=observation or None,
                    done=bool(done),
                )
            )
        step_history.append(per_turn_steps)

    def _record_finished_trajectories(
        self,
        trajectory_turns: list[int],
        dones: list[int],
        *,
        turn_value: int,
    ) -> None:
        for batch_index, done in enumerate(dones):
            if trajectory_turns[batch_index] == 0 and done == 1:
                trajectory_turns[batch_index] = turn_value

    def _finalize_agent_loop_meta(self, state: AgentLoopState) -> dict[str, Any]:
        meta_info = dict(state.meta_info)
        meta_info["turns_stats"] = state.turns_stats.tolist()
        meta_info["active_mask"] = state.active_mask.tolist()
        meta_info["valid_action_stats"] = state.valid_action_stats.tolist()
        meta_info["valid_search_stats"] = state.valid_search_stats.tolist()
        meta_info["trajectory_turns"] = list(state.trajectory_turns)

        # Per-trajectory final answers and timeout flags for reward computation.
        batch_size = len(state.trajectory_turns)
        final_answers = meta_info.get("final_answers", {})
        meta_info["final_answers"] = [
            final_answers.get(idx) for idx in range(batch_size)
        ]
        meta_info["finished_without_answer"] = [
            final_answers.get(idx) is None for idx in range(batch_size)
        ]
        return meta_info

    def _record_policy_rollout_step(
        self,
        state: AgentLoopState,
        parsed_actions: list[PolicyAction],
    ) -> None:
        action_tags = [action.tag for action in parsed_actions]
        state.meta_info.setdefault("policy_action_history", []).append(action_tags)
        state.meta_info.setdefault("policy_action_contents", []).append(
            [action.content for action in parsed_actions]
        )
        if "first_rollout_actions" not in state.meta_info:
            state.meta_info["first_rollout_actions"] = list(action_tags)

    def _record_search_tool_calls(
        self,
        state: AgentLoopState,
        parsed_actions: list[PolicyAction],
    ) -> None:
        search_queries = [
            action.content for action in parsed_actions if action.tag == "search"
        ]
        if not search_queries:
            return

        query_history = state.meta_info.setdefault("search_query_history", [])
        query_history.append(list(search_queries))

        # Per-trajectory query counts — keyed by batch index so that "cats"
        # from trajectory 0 and "cats" from trajectory 1 are tracked separately.
        per_traj: dict[int, dict[str, int]] = state.meta_info.setdefault(
            "per_trajectory_query_counts", {}
        )
        active_list = state.active_mask.tolist()
        for batch_idx, (action, is_active) in enumerate(
            zip(parsed_actions, active_list)
        ):
            if is_active and action.tag == "search":
                traj_counts = per_traj.setdefault(batch_idx, {})
                traj_counts[action.content] = traj_counts.get(action.content, 0) + 1

        flattened_queries = [
            query
            for query_turn in state.meta_info["search_query_history"]
            for query in query_turn
        ]
        state.meta_info["search_queries_total"] = len(flattened_queries)
        state.meta_info["search_queries_unique"] = len(dict.fromkeys(flattened_queries))
        state.meta_info["search_query_repetitions"] = (
            state.meta_info["search_queries_total"]
            - state.meta_info["search_queries_unique"]
        )
        state.meta_info["search_query_reformulations"] = max(
            0,
            state.meta_info["search_queries_unique"] - 1,
        )

    def _record_fetch_tool_calls(
        self,
        state: AgentLoopState,
        parsed_actions: list[PolicyAction],
    ) -> None:
        fetch_urls = [
            [url.strip() for url in re.split(r"[\n,]+", action.content) if url.strip()]
            for action in parsed_actions
            if action.tag == "fetch"
        ]
        if not fetch_urls:
            return

        state.meta_info.setdefault("fetch_url_history", []).extend(fetch_urls)
        state.meta_info["fetched_urls_total"] = sum(
            len(urls) for urls in state.meta_info["fetch_url_history"]
        )

    def _extract_trajectory_answers(
        self,
        state: AgentLoopState,
        step_result: AgentLoopStepResult,
    ) -> None:
        """Store the <answer> content for each trajectory that just terminated.

        Called after every turn so reward computation can read
        meta_info["final_answers"][batch_idx] without re-parsing responses.
        Trajectories that time out without producing <answer> are absent from
        the dict — meta_info["finished_without_answer"] flags them explicitly.
        """
        for batch_idx, (action, done) in enumerate(
            zip(step_result.parsed_actions, step_result.dones)
        ):
            if done and action.tag == "answer":
                state.meta_info.setdefault("final_answers", {})[batch_idx] = (
                    action.content.strip()
                )

    # Stop tool-use turns when the rolling context reaches this fraction of
    # max_prompt_length.  Generating another search + long observation on top of
    # a nearly-full context would be truncated at the input, losing evidence the
    # model already gathered.  Force an answer turn instead.
    _CONTEXT_BUDGET_RATIO: float = 0.85

    def _make_continuation_decision(
        self,
        state: AgentLoopState,
        *,
        turn_index: int,
        allow_tool_use: bool,
    ) -> ContinuationDecision:
        """Decide whether another generation turn should run."""
        active_trajectories = int(state.active_mask.sum().item())
        if active_trajectories <= 0:
            return ContinuationDecision(
                turn_index=turn_index,
                continue_generation=False,
                active_trajectories=0,
                reason="all trajectories have terminated with <answer> or invalid completion",
            )
        if allow_tool_use:
            # Stop early if the context is almost full.  Running another search
            # turn would append max_obs_length more tokens; if the context is
            # already near max_prompt_length the beginning would be truncated,
            # losing evidence that was already retrieved.
            context_lengths = state.meta_info.get("context_token_lengths", [])
            if context_lengths:
                current_len = context_lengths[-1]
                budget = int(self.config.max_prompt_length * self._CONTEXT_BUDGET_RATIO)
                if current_len >= budget:
                    return ContinuationDecision(
                        turn_index=turn_index,
                        continue_generation=False,
                        active_trajectories=active_trajectories,
                        reason=(
                            f"context length {current_len} reached "
                            f"{self._CONTEXT_BUDGET_RATIO:.0%} of max_prompt_length "
                            f"({self.config.max_prompt_length}); "
                            "forcing answer to avoid truncating retrieved evidence"
                        ),
                    )
            return ContinuationDecision(
                turn_index=turn_index,
                continue_generation=True,
                active_trajectories=active_trajectories,
                reason="no <answer> yet for all active trajectories; continue with another agent turn",
            )
        return ContinuationDecision(
            turn_index=turn_index,
            continue_generation=False,
            active_trajectories=active_trajectories,
            reason="search budget exhausted; run final no-tool answer attempt instead",
        )

    def _record_continuation_decision(
        self,
        state: AgentLoopState,
        decision: ContinuationDecision,
    ) -> None:
        state.meta_info.setdefault("continuation_history", []).append(decision)

    def _run_one_turn(
        self,
        gen_batch: SearchBatch,
        state: AgentLoopState,
        *,
        search_mode: str,
        current_step: int,
        total_steps: int,
        do_search: bool,
        include_observations: bool,
    ) -> AgentLoopStepResult:
        """One full turn: trim → generate → execute tool → apply observations."""
        gt_threshold = self.dynamic_threshold(current_step, total_steps)
        state.rollings = self._trim_rollings_for_generation(state.rollings)
        state_snapshots = self._snapshot_rollout_states(state.rollings)
        step_result, step_meta_info = self._run_active_generation_step(
            gen_batch,
            state.rollings,
            search_mode=search_mode,
            gt_threshold=gt_threshold,
            active_mask=state.active_mask,
            do_search=do_search,
            search_cache=state.search_cache,
        )
        state.meta_info.update(step_meta_info)
        self._record_policy_rollout_step(state, step_result.parsed_actions)
        self._record_search_tool_calls(state, step_result.parsed_actions)
        self._record_fetch_tool_calls(state, step_result.parsed_actions)
        self._record_search_steps(state, step_result, state_snapshots)
        self._apply_step_result(
            state, step_result, include_observations=include_observations
        )
        self._extract_trajectory_answers(state, step_result)
        return step_result

    def run_llm_loop(
        self,
        gen_batch: SearchBatch,
        search_mode: str,
        current_step: int = 0,
        total_steps: int = 1,
        initial_input_ids: torch.Tensor | None = None,
    ) -> tuple[SearchBatch, list[int]]:
        """Run the core agent loop: generate → maybe search → append context → repeat."""
        resolved_initial_input_ids = initial_input_ids
        if resolved_initial_input_ids is None:
            resolved_initial_input_ids = gen_batch.batch["input_ids"]
        state = self._initialize_agent_loop_state(gen_batch, resolved_initial_input_ids)
        if "question" in gen_batch.non_tensor_batch:
            state.meta_info["questions"] = list(gen_batch.non_tensor_batch["question"])

        for turn in range(self.config.max_turns):
            decision = self._make_continuation_decision(
                state,
                turn_index=turn,
                allow_tool_use=True,
            )
            self._record_continuation_decision(state, decision)
            if not decision.continue_generation:
                break
            step_result = self._run_one_turn(
                gen_batch,
                state,
                search_mode=search_mode,
                current_step=current_step,
                total_steps=total_steps,
                do_search=True,
                include_observations=True,
            )
            self._record_finished_trajectories(
                state.trajectory_turns, step_result.dones, turn_value=turn + 1
            )

        final_decision = self._make_continuation_decision(
            state,
            turn_index=self.config.max_turns,
            allow_tool_use=False,
        )
        self._record_continuation_decision(state, final_decision)
        if final_decision.active_trajectories > 0:
            step_result = self._run_one_turn(
                gen_batch,
                state,
                search_mode=search_mode,
                current_step=current_step,
                total_steps=total_steps,
                do_search=False,
                include_observations=False,
            )
            self._record_finished_trajectories(
                state.trajectory_turns,
                [1] * len(step_result.dones),
                turn_value=self.config.max_turns + 1,
            )

        state.meta_info = self._finalize_agent_loop_meta(state)
        logger.info("active_traj_num: %s", state.active_num_list)
        traj_tensor = torch.tensor(state.trajectory_turns)
        for turn in range(1, self.config.max_turns + 2):
            logger.info("finish at turn %d: %d", turn, int((traj_tensor == turn).sum()))

        return (
            self._compose_final_output(
                state.original_left_side,
                state.original_right_side,
                state.meta_info,
            ),
            state.trajectory_turns,
        )

    def run_agent_loop(
        self,
        gen_batch: SearchBatch,
        search_mode: str,
        current_step: int = 0,
        total_steps: int = 1,
        initial_input_ids: torch.Tensor | None = None,
    ) -> tuple[SearchBatch, list[int]]:
        """Alias for the core agentic orchestration loop.

        The control flow is explicitly multi-turn:
        `generate -> maybe_call_tool/search -> append_context -> repeat`.
        """

        return self.run_llm_loop(
            gen_batch=gen_batch,
            search_mode=search_mode,
            current_step=current_step,
            total_steps=total_steps,
            initial_input_ids=initial_input_ids,
        )

    def run_prompt_rollout_batch(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[SearchBatch, list[int]]:
        """Run the agent loop directly from a prompt-only training batch.

        This is the high-level RL entrypoint for prompt-only training data:

            PromptBatch -> SearchBatch(gen_batch) -> run_llm_loop()

        The prompt batch contains only prompt-side supervision such as
        `question`, `ground_truth`, and available tools. The model then
        generates its own search / reasoning / stopping / answer trajectory
        inside the loop.
        """
        from src.agent_loop.data import prompt_batch_to_search_batch

        gen_batch = prompt_batch_to_search_batch(prompt_batch)
        return self.run_llm_loop(
            gen_batch=gen_batch,
            search_mode=search_mode,
            current_step=current_step,
            total_steps=total_steps,
        )

    def run_prompt_rollout_group(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        num_rollouts: int = 4,
        sampling_variants: list[dict[str, Any]] | None = None,
        group_id: str | None = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> list[GroupedRolloutBatch]:
        """Sample multiple trajectories for the same prompt group.

        This is the GRPO rollout primitive on the training-side agent loop:

            same prompt -> rollout 1
                        -> rollout 2
                        -> rollout 3
                        -> rollout 4

        Each rollout gets the same ``group_id`` plus its own
        ``rollout_index`` and ``sampling_params`` metadata.  When
        ``base_seed`` is provided, rollout ``i`` receives seed
        ``base_seed + i`` unless that sampling variant already specifies a
        seed explicitly.

        The manager currently applies ``temperature`` overrides directly to
        ``GenerationConfig.temperature`` and seeds Python/Torch RNGs for the
        duration of each rollout.  Other sampling params are preserved as
        metadata on the returned rollout batch so custom backends can consume
        them if needed.
        """
        from src.agent_loop.data import prompt_batch_to_search_batch
        from src.agent_loop.grpo import build_grpo_sampling_params

        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive.")

        resolved_group_id = group_id or f"prompt_group_{uuid4().hex}"
        variants = sampling_variants or build_grpo_sampling_params(
            sampling_params,
            num_rollouts=num_rollouts,
        )
        if len(variants) != num_rollouts:
            raise ValueError("sampling_variants length must equal num_rollouts.")

        grouped_rollouts: list[GroupedRolloutBatch] = []
        for rollout_index, variant in enumerate(variants):
            resolved_variant = dict(variant)
            if base_seed is not None and "seed" not in resolved_variant:
                resolved_variant["seed"] = base_seed + rollout_index

            py_state = random.getstate()
            torch_state = torch.random.get_rng_state()
            old_temperature = float(self.config.temperature)

            try:
                seed = resolved_variant.get("seed")
                if seed is not None:
                    random.seed(int(seed))
                    torch.manual_seed(int(seed))

                if "temperature" in resolved_variant:
                    self.config = dataclass_replace(
                        self.config,
                        temperature=float(resolved_variant["temperature"]),
                    )

                gen_batch = prompt_batch_to_search_batch(prompt_batch)
                gen_batch.non_tensor_batch["sampling_params"] = dict(resolved_variant)
                final_batch, _ = self.run_llm_loop(
                    gen_batch=gen_batch,
                    search_mode=search_mode,
                    current_step=current_step,
                    total_steps=total_steps,
                )
            finally:
                self.config = dataclass_replace(
                    self.config, temperature=old_temperature
                )
                random.setstate(py_state)
                torch.random.set_rng_state(torch_state)

            final_batch.meta_info["group_id"] = resolved_group_id
            final_batch.meta_info["rollout_index"] = rollout_index
            final_batch.meta_info["sampling_params"] = dict(resolved_variant)
            final_output = self.build_final_gen_batch_output(final_batch)
            final_batch.non_tensor_batch["final_gen_batch_output"] = final_output
            grouped_rollouts.append(
                GroupedRolloutBatch(
                    group_id=resolved_group_id,
                    rollout_index=rollout_index,
                    sampling_params=dict(resolved_variant),
                    final_output=final_output,
                )
            )

        return grouped_rollouts

    def build_rollout_outputs(self, batch: SearchBatch) -> list[AgentLoopOutput]:
        """Convert a run_llm_loop SearchBatch to per-trajectory AgentLoopOutputs.

        Bridges the training loop output to the reward-computation interface:

            run_llm_loop() -> SearchBatch -> build_rollout_outputs()
                                                     |
                                                     v
                                              list[AgentLoopOutput]
                                                     |
                                                     v
                                          SearchRewardFunction.compute()

        Per-trajectory metrics are derived from the ``RolloutTrajectory`` steps
        already embedded in ``batch.non_tensor_batch["trajectories"]``, so no
        extra state is required.

        Metrics populated per trajectory:
            rounds_used                           — number of <search> actions
            repeated_search_queries               — duplicate search queries
            fetched_pages                         — number of <fetch> actions
            search_budget_exhausted_without_answer — 1.0 if timed out without answering
            subquestion_coverage_ratio            — 1.0 (not tracked in training loop)
        """
        from src.agent_loop.agent_loop import AgentLoopOutput

        pad_id: int = getattr(self.tokenizer, "pad_token_id", 0) or 0
        trajectories: list[RolloutTrajectory] = batch.non_tensor_batch.get(
            "trajectories", []
        )
        group_id = batch.meta_info.get("group_id")
        rollout_index = batch.meta_info.get("rollout_index")
        valid_search_stats: list[int] = batch.meta_info.get("valid_search_stats", [])
        # Per-trajectory query counts recorded by _record_search_tool_calls.
        # Using this avoids scanning traj.steps and gives correct counts because
        # each trajectory's queries are tracked separately.
        per_traj_counts: dict[int, dict[str, int]] = batch.meta_info.get(
            "per_trajectory_query_counts", {}
        )

        outputs: list[AgentLoopOutput] = []
        for i, traj in enumerate(trajectories):
            response_mask = [
                1 if tid != pad_id else 0 for tid in traj.response_with_observation_mask
            ]

            traj_query_counts = per_traj_counts.get(i, {})
            repeated = sum(
                count - 1 for count in traj_query_counts.values() if count > 1
            )
            fetch_count = sum(1 for step in traj.steps if step.action_tag == "fetch")
            rounds = (
                float(valid_search_stats[i]) if i < len(valid_search_stats) else 0.0
            )

            outputs.append(
                AgentLoopOutput(
                    prompt_ids=list(traj.prompt_token_ids),
                    response_ids=list(traj.response_token_ids),
                    response_mask=response_mask,
                    num_turns=traj.trajectory_turns,
                    metrics={
                        "rounds_used": rounds,
                        "search_rounds": rounds,
                        "repeated_search_queries": float(repeated),
                        "fetched_pages": float(fetch_count),
                        "subquestion_coverage_ratio": 1.0,
                        "repeated_query_ratio": (
                            repeated / max(sum(traj_query_counts.values()), 1)
                        ),
                        "search_budget_exhausted_without_answer": float(
                            traj.finished_without_answer
                        ),
                    },
                    final_answer=traj.final_answer,
                    group_id=group_id,
                    rollout_index=rollout_index,
                )
            )
        return outputs

    def build_search_trajectory_logs(
        self,
        batch: SearchBatch,
    ) -> list[SearchTrajectoryLog]:
        """Build step-by-step printable trajectory logs for each prompt."""
        step_history: list[list[SearchStep | None]] = batch.meta_info.get(
            "search_step_history", []
        )
        questions: list[str] = batch.non_tensor_batch.get("question", [])
        final_answers: list[str | None] = batch.meta_info.get("final_answers", [])
        finished_without_answer: list[bool] = batch.meta_info.get(
            "finished_without_answer", []
        )
        batch_size = len(questions) if questions else len(final_answers)
        if batch_size == 0:
            trajectories: list[RolloutTrajectory] = batch.non_tensor_batch.get(
                "trajectories", []
            )
            batch_size = len(trajectories)
        logs: list[SearchTrajectoryLog] = []
        for batch_index in range(batch_size):
            steps = [
                step
                for turn_steps in step_history
                if batch_index < len(turn_steps)
                for step in (turn_steps[batch_index],)
                if step is not None
            ]
            logs.append(
                SearchTrajectoryLog(
                    batch_index=batch_index,
                    question=questions[batch_index]
                    if batch_index < len(questions)
                    else "",
                    steps=steps,
                    final_answer=final_answers[batch_index]
                    if batch_index < len(final_answers)
                    else None,
                    finished_without_answer=finished_without_answer[batch_index]
                    if batch_index < len(finished_without_answer)
                    else True,
                )
            )
        return logs

    def build_final_gen_batch_output(self, batch: SearchBatch) -> FinalGenBatchOutput:
        """Build the unified final rollout object for one generated batch.

        Consolidates the tensor batch, per-trajectory symbolic traces, and
        reward-ready AgentLoopOutput objects into a single ``FinalGenBatchOutput``
        that contains everything a GRPO/PPO training step needs.

        Call this after ``run_llm_loop`` / ``run_prompt_rollout_batch``::

            batch, _ = manager.run_prompt_rollout_batch(prompt_batch, ...)
            final = manager.build_final_gen_batch_output(batch)
        """
        trajectories: list[RolloutTrajectory] = batch.non_tensor_batch.get(
            "trajectories", []
        )
        trajectory_logs = self.build_search_trajectory_logs(batch)
        rollout_outputs = self.build_rollout_outputs(batch)
        trajectory_turns: list[int] = batch.meta_info.get("trajectory_turns", [])
        return FinalGenBatchOutput(
            search_batch=batch,
            trajectories=trajectories,
            trajectory_logs=trajectory_logs,
            rollout_outputs=rollout_outputs,
            trajectory_turns=trajectory_turns,
        )

    def _compose_final_output(
        self,
        left_side: dict[str, torch.Tensor],
        right_side: dict[str, torch.Tensor],
        meta_info: dict[str, Any],
    ) -> SearchBatch:
        """Assemble the training-ready RL rollout batch.

        Tensor layout (all in search_batch.batch):
            input_ids               — prompt + full response (prompt ++ actions ++ observations)
            attention_mask          — which token positions are non-pad
            info_mask               — 1 for model-generated tokens, 0 for env-injected observations
            position_ids            — incremental positions over attention_mask
            prompts                 — prompt token ids only (left side)
            responses               — response token ids only (right side)
            responses_with_info_mask — responses with observation tokens zeroed (for loss masking)

        Non-tensor data (search_batch.non_tensor_batch):
            trajectories — list[RolloutTrajectory], one per prompt in the batch
        """
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]
        final_output["input_ids"] = torch.cat(
            [left_side["input_ids"], right_side["responses"]], dim=1
        )
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(
                    final_output["responses_with_info_mask"]
                ),
            ],
            dim=1,
        )
        final_output["position_ids"] = self.tensor_fn.create_position_ids(
            final_output["attention_mask"]
        )

        search_batch = SearchBatch.from_dict(final_output)
        search_batch.meta_info.update(meta_info)
        if "questions" in meta_info:
            search_batch.non_tensor_batch["question"] = list(meta_info["questions"])
        search_batch.non_tensor_batch["trajectories"] = (
            self._build_rollout_trajectories(
                left_side=left_side, right_side=right_side, meta_info=meta_info
            )
        )
        search_batch.non_tensor_batch["trajectory_logs"] = (
            self.build_search_trajectory_logs(search_batch)
        )
        if isinstance(self.generation_backend, LogProbCapable):
            self.compute_log_prob(search_batch, store_key="old_log_probs")
        return search_batch

    def _build_rollout_trajectories(
        self,
        *,
        left_side: dict[str, torch.Tensor],
        right_side: dict[str, torch.Tensor],
        meta_info: dict[str, Any],
    ) -> list[RolloutTrajectory]:
        """Build one RolloutTrajectory per prompt from the completed loop state.

        Each trajectory captures the full RL rollout:
            prompt → search → retrieval → reasoning → answer

        Fields are structured for reward computation and SFT data extraction
        without re-parsing the raw token sequences.
        """
        prompt_ids = left_side["input_ids"].tolist()
        response_ids = right_side["responses"].tolist()
        response_with_obs = right_side["responses_with_info_mask"].tolist()
        prompt_attention_mask = self.tensor_fn.create_attention_mask(
            left_side["input_ids"]
        ).tolist()
        response_attention_mask = self.tensor_fn.create_attention_mask(
            right_side["responses"]
        ).tolist()
        response_action_mask = self.tensor_fn.create_attention_mask(
            right_side["responses_with_info_mask"]
        ).tolist()

        react_trajectory = meta_info.get("react_trajectory", [])
        trajectory_turns = meta_info.get("trajectory_turns", [0] * len(prompt_ids))
        final_answers = meta_info.get("final_answers", [])
        finished_without_answer = meta_info.get("finished_without_answer", [])

        trajectories: list[RolloutTrajectory] = []
        for batch_index in range(len(prompt_ids)):
            steps = [
                step
                for turn_steps in react_trajectory
                if batch_index < len(turn_steps)
                # None marks a turn where this trajectory was already done and
                # did not participate — filter those placeholders out.
                for step in (turn_steps[batch_index],)
                if step is not None
            ]
            trajectories.append(
                RolloutTrajectory(
                    batch_index=batch_index,
                    prompt_token_ids=list(prompt_ids[batch_index]),
                    response_token_ids=list(response_ids[batch_index]),
                    response_with_observation_mask=list(response_with_obs[batch_index]),
                    trajectory_turns=trajectory_turns[batch_index]
                    if batch_index < len(trajectory_turns)
                    else 0,
                    steps=steps,
                    final_answer=final_answers[batch_index]
                    if batch_index < len(final_answers)
                    else None,
                    finished_without_answer=finished_without_answer[batch_index]
                    if batch_index < len(finished_without_answer)
                    else True,
                    tokens=list(prompt_ids[batch_index])
                    + list(response_ids[batch_index]),
                    attention_mask=list(prompt_attention_mask[batch_index])
                    + list(response_attention_mask[batch_index]),
                    response_mask=([0] * len(prompt_ids[batch_index]))
                    + list(response_action_mask[batch_index]),
                )
            )
        return trajectories

    def _cached_batch_search(
        self,
        tool_calls: list[SearchToolCall],
        search_mode: str,
        gt_threshold: float,
        cache: dict[tuple[int, str], str],
    ) -> list[str]:
        """Call the retriever only for queries not already in the rollout cache.

        Repeated queries within the same trajectory return the cached result
        immediately, avoiding redundant retriever calls and making the per-
        trajectory ``repeated_search_queries`` metric accurate.
        """
        results: list[str | None] = [None] * len(tool_calls)
        uncached_calls: list[SearchToolCall] = []
        uncached_positions: list[int] = []
        for pos, call in enumerate(tool_calls):
            key = (call.batch_index, call.query)
            if key in cache:
                results[pos] = cache[key]
            else:
                uncached_calls.append(call)
                uncached_positions.append(pos)
        if uncached_calls:
            new_results = self.batch_search(uncached_calls, search_mode, gt_threshold)
            for pos, result, call in zip(
                uncached_positions, new_results, uncached_calls
            ):
                cache[(call.batch_index, call.query)] = result
                results[pos] = result
        return [r for r in results if r is not None]

    def execute_predictions(
        self,
        predictions: list[str],
        problem: list[str],
        ground_truth: list[Any],
        search_mode: str,
        gt_threshold: float,
        active_mask: torch.Tensor | None = None,
        do_search: bool = True,
        search_cache: dict[tuple[int, str], str] | None = None,
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        sampled_actions = self.parse_policy_actions(predictions)
        batch_size = len(predictions)
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool)

        search_tool_calls = self.build_search_tool_calls(
            sampled_actions,
            problem=problem,
            ground_truth=ground_truth,
            active_mask=active_mask,
        )
        fetch_tool_calls = self.build_fetch_tool_calls(
            sampled_actions,
            active_mask=active_mask,
        )

        if do_search and search_tool_calls:
            if search_cache is not None:
                search_results = self._cached_batch_search(
                    search_tool_calls, search_mode, gt_threshold, search_cache
                )
            else:
                search_results = self.batch_search(
                    search_tool_calls, search_mode, gt_threshold
                )
        else:
            search_results = [""] * len(search_tool_calls)
        if do_search and fetch_tool_calls:
            fetch_results = self.batch_fetch(fetch_tool_calls)
        else:
            fetch_results = [""] * len(fetch_tool_calls)

        next_obs: list[str] = []
        dones: list[int] = []
        valid_action: list[int] = []
        is_search: list[int] = []
        search_result_index = 0
        fetch_result_index = 0

        for sample_action, active in zip(sampled_actions, active_mask.tolist()):
            tag = sample_action.tag
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            is_done = tag == "answer"
            retrieval = search_results[search_result_index] if tag == "search" else ""
            fetch_content = fetch_results[fetch_result_index] if tag == "fetch" else ""
            if tag == "search":
                search_result_index += 1
            if tag == "fetch":
                fetch_result_index += 1

            next_obs.append(
                ""
                if is_done
                else build_react_observation(
                    sample_action,
                    retrieval if tag == "search" else fetch_content,
                )
            )
            dones.append(1 if is_done else 0)
            valid_action.append(
                1 if tag in {"answer", "search", "plan", "fetch"} else 0
            )
            is_search.append(1 if tag == "search" else 0)

        return next_obs, dones, valid_action, is_search

    def dynamic_threshold(
        self,
        current_step: int,
        total_steps: int,
    ) -> float:
        if current_step >= total_steps:
            return self.config.end_threshold

        progress = current_step / total_steps
        exp_base = getattr(self.config, "exp_base", 4)
        exp_value = (math.pow(exp_base, progress) - 1) / (exp_base - 1)
        return self.config.start_threshold + exp_value * (
            self.config.end_threshold - self.config.start_threshold
        )

    def parse_policy_actions(self, predictions: list[Any]) -> list[PolicyAction]:
        """Parse one sampled policy action per trajectory step."""

        actions: list[PolicyAction] = []
        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            match = ACTION_PATTERN.search(prediction)
            if match:
                actions.append(
                    PolicyAction(
                        tag=match.group("tag"),
                        content=match.group("content").strip(),
                        raw_text=prediction,
                    )
                )
            else:
                actions.append(PolicyAction(tag=None, content="", raw_text=prediction))
        return actions

    def build_search_tool_calls(
        self,
        parsed_actions: list[PolicyAction],
        *,
        problem: list[str],
        ground_truth: list[Any],
        active_mask: torch.Tensor,
    ) -> list[SearchToolCall]:
        """Convert model-emitted search actions into explicit tool calls."""
        return [
            SearchToolCall(
                query=decision.query,
                problem=decision.problem,
                ground_truth=decision.ground_truth,
                batch_index=decision.batch_index,
            )
            for decision in self.build_search_decisions(
                parsed_actions,
                problem=problem,
                ground_truth=ground_truth,
                active_mask=active_mask,
            )
        ]

    def build_search_decisions(
        self,
        parsed_actions: list[PolicyAction],
        *,
        problem: list[str],
        ground_truth: list[Any],
        active_mask: torch.Tensor,
    ) -> list[SearchDecision]:
        """Convert model-emitted `<search>` actions into explicit decisions.

        This is the moment where the policy decides to call the search tool,
        rather than the environment hardcoding retrieval. Query reformulation
        and multi-hop search naturally fall out of repeated search decisions
        across turns.
        """

        decisions: list[SearchDecision] = []
        for index, (action, active) in enumerate(
            zip(parsed_actions, active_mask.tolist())
        ):
            if not active or action.tag != "search":
                continue
            query = action.content.strip()
            if not query:
                continue
            decisions.append(
                SearchDecision(
                    query=query,
                    problem=problem[index],
                    ground_truth=_resolve_ground_truth_text(ground_truth[index]),
                    batch_index=index,
                    raw_action_text=action.raw_text,
                )
            )
        return decisions

    def build_fetch_tool_calls(
        self,
        parsed_actions: list[PolicyAction],
        *,
        active_mask: torch.Tensor,
    ) -> list[FetchToolCall]:
        """Convert model-emitted fetch actions into explicit tool calls."""

        calls: list[FetchToolCall] = []
        for index, (action, active) in enumerate(
            zip(parsed_actions, active_mask.tolist())
        ):
            if not active or action.tag != "fetch":
                continue
            urls = [
                part.strip()
                for part in re.split(r"[\n,]+", action.content)
                if part.strip()
            ]
            calls.append(FetchToolCall(urls=urls, batch_index=index))
        return calls

    def _resolve_fetcher(self) -> Fetcher | None:
        if self.fetcher is not None:
            return self.fetcher
        if self.config.retrieval_url:
            return EndpointFetcher(
                _derive_fetch_url(self.config.retrieval_url),
                retry_attempts=self.config.wiki_retry_attempts,
                sleep_seconds=self.config.wiki_retry_sleep_seconds,
            )
        return None

    def postprocess_predictions(
        self, predictions: list[Any]
    ) -> tuple[list[str | None], list[str]]:
        parsed_actions = self.parse_policy_actions(predictions)
        return [action.tag for action in parsed_actions], [
            action.content for action in parsed_actions
        ]

    def _make_retriever(
        self,
        tool_call: SearchToolCall,
        search_mode: str,
        gt_threshold: float,
    ) -> Retriever:
        """Return the backend Retriever for this search mode.

        Adding a new backend (Tavily, Bing, …) only requires adding one
        branch here and implementing the Retriever protocol.
        """
        cfg = self.config
        if search_mode == "google":
            return GoogleRetriever(
                os.environ.get("SERP_API_KEY", ""),
                cfg.google_retry_attempts,
                cfg.google_retry_sleep_seconds,
            )
        if search_mode == "wiki":
            url = f"http://{cfg.retriever_ip}:6002/retrieve" if cfg.retriever_ip else ""
            return EndpointRetriever(
                url, cfg.wiki_retry_attempts, cfg.wiki_retry_sleep_seconds
            )
        if search_mode == "local":
            return EndpointRetriever(
                cfg.retrieval_url or "",
                cfg.wiki_retry_attempts,
                cfg.wiki_retry_sleep_seconds,
            )
        if search_mode in ("simulate_sft", "simulate_prompt"):
            return SimulateRetriever(
                cfg.llm_ip,
                cfg.temperature,
                cfg.llm_max_retries,
                problem=tool_call.problem,
                ground_truth=tool_call.ground_truth,
                gt_threshold=gt_threshold,
                use_examples=(search_mode == "simulate_prompt"),
            )
        return EndpointRetriever("", 0, 0.0)  # unknown mode → no-op

    def _endpoint_batch_search(
        self,
        tool_calls: list[SearchToolCall],
        search_mode: str,
    ) -> list[str]:
        """Send all queries in one POST /retrieve request.

        Reduces N parallel HTTP requests to 1 for endpoint-based backends
        (``local`` and ``wiki`` modes).  Results are aligned to *tool_calls*
        by the server's ordered response array.
        """
        cfg = self.config
        if search_mode == "wiki":
            url = f"http://{cfg.retriever_ip}:6002/retrieve" if cfg.retriever_ip else ""
        else:
            url = cfg.retrieval_url or ""
        retriever = EndpointRetriever(
            url, cfg.wiki_retry_attempts, cfg.wiki_retry_sleep_seconds
        )
        per_query_docs = retriever.retrieve_batch(
            [tc.query for tc in tool_calls], cfg.topk
        )
        return [_format_documents(docs) or _NO_INFO for docs in per_query_docs]

    def batch_search(
        self,
        tool_calls: list[SearchToolCall],
        search_mode: str,
        gt_threshold: float,
    ) -> list[str]:
        if not tool_calls:
            return []
        # Fast path: one HTTP request for all queries on endpoint-based backends.
        if search_mode in ("local", "wiki"):
            return self._endpoint_batch_search(tool_calls, search_mode)
        # Standard path: parallel per-query threads for Google and simulate modes.
        all_results = [_NO_INFO] * len(tool_calls)
        max_workers = min(self.config.search_max_workers, len(tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._search, tc, search_mode, gt_threshold): pos
                for pos, tc in enumerate(tool_calls)
            }
            for future in as_completed(futures):
                try:
                    result, _ = future.result()
                    all_results[futures[future]] = result
                except Exception as exc:
                    logger.warning("Search future failed: %s", exc)
        return all_results

    def batch_fetch(self, tool_calls: list[FetchToolCall]) -> list[str]:
        if not tool_calls:
            return []
        fetcher = self._resolve_fetcher()
        if fetcher is None:
            return [_NO_INFO] * len(tool_calls)

        all_results = [_NO_INFO] * len(tool_calls)
        max_workers = min(self.config.search_max_workers, len(tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch, tc, fetcher): pos
                for pos, tc in enumerate(tool_calls)
            }
            for future in as_completed(futures):
                try:
                    result, _ = future.result()
                    all_results[futures[future]] = result
                except Exception as exc:
                    logger.warning("Fetch future failed: %s", exc)
        return all_results

    def _search(
        self,
        tool_call: SearchToolCall,
        search_mode: str,
        gt_threshold: float,
    ) -> tuple[str, int]:
        retriever = self._make_retriever(tool_call, search_mode, gt_threshold)
        docs = retriever.retrieve(tool_call.query, self.config.topk)
        result = _format_documents(docs)
        return result or _NO_INFO, tool_call.batch_index

    def retrieve_documents(
        self,
        query: str,
        *,
        search_mode: str | None = None,
        problem: str = "",
        ground_truth: str = "",
        gt_threshold: float = 0.0,
        topk: int | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve structured documents from any configured backend.

        This is the normalized retrieval entrypoint for BM25, dense retrieval,
        hybrid search, web search, and internal RAG-style backends.
        """
        tool_call = SearchToolCall(
            query=query,
            problem=problem,
            ground_truth=ground_truth,
            batch_index=0,
        )
        retriever = self._make_retriever(
            tool_call,
            search_mode or self.config.search_mode,
            gt_threshold,
        )
        return retriever.retrieve(query, topk or self.config.topk)

    def _fetch(
        self,
        tool_call: FetchToolCall,
        fetcher: Fetcher,
    ) -> tuple[str, int]:
        result = fetcher.fetch(tool_call.urls)
        return result or _NO_INFO, tool_call.batch_index

    def retrieve_from_wiki(self, ip: str | None, query: str, topk: int = 5) -> str:
        if not ip:
            return _NO_INFO
        return EndpointRetriever(
            f"http://{ip}:6002/retrieve",
            self.config.wiki_retry_attempts,
            self.config.wiki_retry_sleep_seconds,
        ).search(query, topk)

    def retrieve_from_local(self, query: str, topk: int = 5) -> str:
        return EndpointRetriever(
            self.config.retrieval_url or "",
            self.config.wiki_retry_attempts,
            self.config.wiki_retry_sleep_seconds,
        ).search(query, topk)

    def retrieve_from_google(
        self, query: str, topk: int, retry_attempt: int | None = None
    ) -> str:
        return GoogleRetriever(
            os.environ.get("SERP_API_KEY", ""),
            retry_attempt or self.config.google_retry_attempts,
            self.config.google_retry_sleep_seconds,
        ).search(query, topk)

    def _passages2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        return _format_documents([_parse_api_item(item) for item in retrieval_result])
