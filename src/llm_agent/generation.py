"""Generation manager for search-oriented, tool-using LLM loops."""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

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


ROLLOUT_ACTION_TAGS = ("plan", "search", "fetch", "answer")
ACTION_PATTERN = re.compile(
    rf"<(?P<tag>{'|'.join(ROLLOUT_ACTION_TAGS)})>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL,
)

_NO_INFO = "No information available"


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
    ) -> None:
        self.tokenizer = tokenizer
        self.generation_backend = generation_backend or actor_rollout_wg
        if self.generation_backend is None:
            raise ValueError("generation_backend is required.")
        self.config = config
        self.is_validation = is_validation
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

    def _process_next_obs(self, next_obs: list[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]
        return next_obs_ids

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
    ) -> tuple[AgentLoopStepResult, dict[str, Any]]:
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
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            responses_str,
            gen_batch.non_tensor_batch["question"],
            gen_batch.non_tensor_batch["golden_answers"],
            search_mode,
            gt_threshold,
            active_mask,
            do_search=do_search,
        )
        parsed_actions = self.parse_policy_actions(responses_str)
        return (
            AgentLoopStepResult(
                responses_ids=responses_ids,
                responses_str=responses_str,
                parsed_actions=parsed_actions,
                next_obs=next_obs,
                dones=dones,
                valid_action=valid_action,
                is_search=is_search,
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
        curr_active_mask = torch.tensor(
            [not done for done in step_result.dones], dtype=torch.bool
        )
        state.active_mask = state.active_mask & curr_active_mask
        state.active_num_list.append(int(state.active_mask.sum().item()))

        if include_observations:
            state.turns_stats[curr_active_mask] += 1
            next_obs_ids = self._process_next_obs(step_result.next_obs)
            state.rollings = self._update_rolling_state(
                state.rollings,
                step_result.responses_ids,
                next_obs_ids,
            )
            state.original_right_side = self._update_right_side(
                state.original_right_side,
                step_result.responses_ids,
                next_obs_ids,
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
        step_result, step_meta_info = self._run_active_generation_step(
            gen_batch,
            state.rollings,
            search_mode=search_mode,
            gt_threshold=gt_threshold,
            active_mask=state.active_mask,
            do_search=do_search,
        )
        state.meta_info.update(step_meta_info)
        self._record_policy_rollout_step(state, step_result.parsed_actions)
        self._record_search_tool_calls(state, step_result.parsed_actions)
        self._apply_step_result(
            state, step_result, include_observations=include_observations
        )
        return step_result

    def run_llm_loop(
        self,
        gen_batch: SearchBatch,
        search_mode: str,
        current_step: int,
        total_steps: int,
        initial_input_ids: torch.Tensor,
    ) -> tuple[SearchBatch, list[int]]:
        """Run the core agent loop: generate → maybe search → append context → repeat."""
        state = self._initialize_agent_loop_state(gen_batch, initial_input_ids)

        for turn in range(self.config.max_turns):
            if not bool(state.active_mask.sum()):
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

        if bool(state.active_mask.sum()):
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
                state.original_left_side, state.original_right_side, state.meta_info
            ),
            state.trajectory_turns,
        )

    def run_agent_loop(
        self,
        gen_batch: SearchBatch,
        search_mode: str,
        current_step: int,
        total_steps: int,
        initial_input_ids: torch.Tensor,
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

    def _compose_final_output(
        self,
        left_side: dict[str, torch.Tensor],
        right_side: dict[str, torch.Tensor],
        meta_info: dict[str, Any],
    ) -> SearchBatch:
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
        return search_batch

    def execute_predictions(
        self,
        predictions: list[str],
        problem: list[str],
        ground_truth: list[Any],
        search_mode: str,
        gt_threshold: float,
        active_mask: torch.Tensor | None = None,
        do_search: bool = True,
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        sampled_actions = self.parse_policy_actions(predictions)
        batch_size = len(predictions)
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool)

        cur_actions = [action.tag for action in sampled_actions]
        search_tool_calls = self.build_search_tool_calls(
            sampled_actions,
            problem=problem,
            ground_truth=ground_truth,
            active_mask=active_mask,
        )

        if do_search and search_tool_calls:
            search_results = self.batch_search(
                [
                    (call.query, call.problem, call.ground_truth)
                    for call in search_tool_calls
                ],
                search_mode,
                gt_threshold,
            )
        else:
            search_results = [""] * len(search_tool_calls)

        next_obs: list[str] = []
        dones: list[int] = []
        valid_action: list[int] = []
        is_search: list[int] = []
        search_result_index = 0

        for action, active in zip(cur_actions, active_mask.tolist()):
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            if action == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                continue

            if action == "search":
                next_obs.append(
                    f"\n\n<information>{search_results[search_result_index].strip()}</information>\n\n"
                )
                search_result_index += 1
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                continue

            if action == "plan":
                next_obs.append(
                    "\n\n<plan_feedback>Plan recorded. Continue with <search>, <fetch>, or <answer>.</plan_feedback>\n\n"
                )
                dones.append(0)
                valid_action.append(1)
                is_search.append(0)
                continue

            if action == "fetch":
                next_obs.append(
                    "\n\n<tool_feedback>Fetch was requested, but this rollout manager does not yet execute fetch actions directly. Use <search> for retrieval-backed evidence in this loop.</tool_feedback>\n\n"
                )
                dones.append(0)
                valid_action.append(1)
                is_search.append(0)
                continue

            next_obs.append(
                "\nMy previous action is invalid. "
                "Valid actions in this rollout loop are <plan>, <search>, <fetch>, and <answer>. "
                "Let me try again.\n"
            )
            dones.append(0)
            valid_action.append(0)
            is_search.append(0)

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

        calls: list[SearchToolCall] = []
        for index, (action, active) in enumerate(
            zip(parsed_actions, active_mask.tolist())
        ):
            if not active or action.tag != "search":
                continue
            calls.append(
                SearchToolCall(
                    query=action.content,
                    problem=problem[index],
                    ground_truth=_resolve_ground_truth_text(ground_truth[index]),
                    batch_index=index,
                )
            )
        return calls

    def postprocess_predictions(
        self, predictions: list[Any]
    ) -> tuple[list[str | None], list[str]]:
        parsed_actions = self.parse_policy_actions(predictions)
        return [action.tag for action in parsed_actions], [
            action.content for action in parsed_actions
        ]

    def batch_search(
        self,
        search_payload: list[tuple[str, str, str]],
        search_mode: str,
        gt_threshold: float,
    ) -> list[str]:
        if not search_payload:
            return []

        all_search_result = [_NO_INFO] * len(search_payload)
        max_workers = min(self.config.search_max_workers, len(search_payload))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._search,
                    query,
                    problem,
                    ground_truth,
                    search_mode,
                    gt_threshold,
                    index,
                )
                for index, (query, problem, ground_truth) in enumerate(search_payload)
            ]
            for future in as_completed(futures):
                try:
                    result, index = future.result()
                    all_search_result[index] = result
                except Exception as exc:
                    logger.warning("Search future failed: %s", exc)
                    continue
        return all_search_result

    def _retrieve_from_endpoint(
        self,
        url: str,
        query: str,
        topk: int,
        retry_attempts: int,
        sleep_seconds: float = 1.0,
    ) -> str:
        import requests  # optional dep; imported here so the class loads without it

        for _ in range(retry_attempts):
            try:
                resp = requests.post(
                    url, json={"queries": [query], "topk": topk}, timeout=10
                )
                resp.raise_for_status()
                rows = resp.json().get("result", [[]])[0]
                return self._passages2string(rows) or _NO_INFO
            except Exception:  # pragma: no cover - network/runtime dependent
                time.sleep(sleep_seconds)
        return _NO_INFO

    def retrieve_from_wiki(self, ip: str | None, query: str, topk: int = 5) -> str:
        if not ip:
            return _NO_INFO
        return self._retrieve_from_endpoint(
            f"http://{ip}:6002/retrieve",
            query,
            topk,
            self.config.wiki_retry_attempts,
            self.config.wiki_retry_sleep_seconds,
        )

    def retrieve_from_local(self, query: str, topk: int = 5) -> str:
        """Call the repo's retrieval_server (POST /retrieve) directly."""
        if not self.config.retrieval_url:
            return _NO_INFO
        return self._retrieve_from_endpoint(
            self.config.retrieval_url,
            query,
            topk,
            self.config.wiki_retry_attempts,
            self.config.wiki_retry_sleep_seconds,
        )

    def retrieve_from_google(
        self, query: str, topk: int, retry_attempt: int | None = None
    ) -> str:
        retry_attempt = retry_attempt or self.config.google_retry_attempts
        api_key = os.environ.get("SERP_API_KEY")
        if not api_key:
            return _NO_INFO

        import serpapi  # optional dep; imported here so the class loads without it

        params = {"engine": "google", "q": query, "api_key": api_key, "num": topk}
        for attempt in range(retry_attempt):
            try:
                search = serpapi.search(params)
                search_result = search.get("organic_results", [])
                search_texts = []
                for item in search_result:
                    text_data = (
                        f"{item.get('title', '')}{item.get('snippet', '')}".strip()
                    )
                    if text_data:
                        search_texts.append(text_data)
                if not search_texts:
                    return _NO_INFO
                return "\n".join(
                    [
                        f"Doc {index + 1}: {doc}"
                        for index, doc in enumerate(search_texts)
                    ]
                )
            except Exception:  # pragma: no cover - network/runtime dependent
                if attempt < retry_attempt - 1:
                    time.sleep(self.config.google_retry_sleep_seconds)
        return _NO_INFO

    def _search(
        self,
        query: str,
        problem: str,
        ground_truth: str,
        search_mode: str,
        gt_threshold: float,
        index: int,
    ) -> tuple[str, int]:
        if search_mode == "google":
            doc_texts = self.retrieve_from_google(query, self.config.topk)
        elif search_mode == "wiki":
            doc_texts = self.retrieve_from_wiki(
                self.config.retriever_ip, query, self.config.topk
            )
        elif search_mode in ("simulate_sft", "simulate_prompt"):
            doc_texts = search_simulate(
                self.config.llm_ip,
                self.config.topk,
                self.config.temperature,
                query,
                problem,
                ground_truth,
                gt_threshold,
                use_examples=(search_mode == "simulate_prompt"),
                llm_max_retries=self.config.llm_max_retries,
            )
        elif search_mode == "local":
            doc_texts = self.retrieve_from_local(query, self.config.topk)
        else:
            doc_texts = "No information available"
        return doc_texts, index

    def _passages2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        parts = []
        for index, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title, _, text = content.partition("\n")
            parts.append(f"Doc {index + 1}(Title: {title}) {text}")
        return "\n".join(parts)
