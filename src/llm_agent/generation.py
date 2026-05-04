"""Generation manager for search-oriented, tool-using LLM loops."""

from __future__ import annotations

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


@dataclass
class SearchBatch:
    """Lightweight batch container for multi-turn search generation."""

    batch: dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_batch: dict[str, Any] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, batch: dict[str, torch.Tensor]) -> "SearchBatch":
        return cls(batch=batch, non_tensor_batch={}, meta_info={})


ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)

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


def ask_llm(ip_list_raw: str | None, prompt: str, temperature: float, max_retries: int = 5) -> str:
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


def search_simulate_sft(
    ip: str | None,
    topk: int,
    temperature: float,
    query: str,
    problem: str,
    ground_truth: str,
    gt_threshold: float,
    llm_max_retries: int = 5,
) -> str:
    prob = random.random()
    if prob > gt_threshold:
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
    return "\n".join(results.replace("\n\n", "\n").split("\n")).split(f"Doc {topk + 1}")[0]


def search_simulate_prompt(
    ip: str | None,
    topk: int,
    temperature: float,
    query: str,
    problem: str,
    ground_truth: str,
    gt_threshold: float,
    llm_max_retries: int = 5,
) -> str:
    prob = random.random()
    if prob > gt_threshold:
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

    results = ask_llm(ip, prompt, temperature, max_retries=llm_max_retries)
    return "\n".join(results.replace("\n\n", "\n").split("\n")).split(f"Doc {topk + 1}")[0]


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

    def _postprocess_responses(self, responses: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        responses_str = [
            response.split("</search>")[0] + "</search>"
            if "</search>" in response
            else response.split("</answer>")[0] + "</answer>"
            if "</answer>" in response
            else response
            for response in responses_str
        ]

        return self._batch_tokenize(responses_str), responses_str

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
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
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
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                pad_to_left=False,
            )

        effective_len = int(self.tensor_fn.create_attention_mask(responses).sum(dim=1).max().item())
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
        trimmed_batch = {key: value[:-padding_size] for key, value in padded_output.batch.items()}
        trimmed_meta = {}
        for key, value in getattr(padded_output, "meta_info", {}).items():
            trimmed_meta[key] = value[:-padding_size] if isinstance(value, torch.Tensor) else value

        padded_output.batch = trimmed_batch
        padded_output.meta_info = trimmed_meta
        return padded_output

    def run_llm_loop(
        self,
        gen_batch: SearchBatch,
        search_mode: str,
        current_step: int,
        total_steps: int,
        initial_input_ids: torch.Tensor,
    ) -> tuple[SearchBatch, list[int]]:
        original_left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length :]}
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }
        batch_size = gen_batch.batch["input_ids"].shape[0]
        trajectory_turns = [0 for _ in range(batch_size)]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [int(active_mask.sum().item())]
        rollings = gen_batch
        meta_info: dict[str, Any] = {}

        for step in range(self.config.max_turns):
            if not bool(active_mask.sum()):
                break

            gt_threshold = self.dynamic_threshold(
                current_step,
                total_steps,
                step + 1,
                self.config.max_turns + 1,
            )
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = SearchBatch.from_dict(
                {key: value[active_mask] for key, value in rollings.batch.items()}
            )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = getattr(gen_output, "meta_info", {}) or {}
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
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
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(int(active_mask.sum().item()))
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

            for batch_index, done in enumerate(dones):
                if trajectory_turns[batch_index] == 0 and done == 1:
                    trajectory_turns[batch_index] = step + 1

        if bool(active_mask.sum()):
            gt_threshold = self.dynamic_threshold(
                current_step,
                total_steps,
                self.config.max_turns + 1,
                self.config.max_turns + 1,
            )
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = SearchBatch.from_dict(
                {key: value[active_mask] for key, value in rollings.batch.items()}
            )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = getattr(gen_output, "meta_info", {}) or {}
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(
                responses_ids,
                responses_str,
                active_mask,
            )
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str,
                gen_batch.non_tensor_batch["question"],
                gen_batch.non_tensor_batch["golden_answers"],
                search_mode,
                gt_threshold,
                active_mask,
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(int(active_mask.sum().item()))
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            original_right_side = self._update_right_side(original_right_side, responses_ids)

            meta_info["turns_stats"] = turns_stats.tolist()
            meta_info["active_mask"] = active_mask.tolist()
            meta_info["valid_action_stats"] = valid_action_stats.tolist()
            meta_info["valid_search_stats"] = valid_search_stats.tolist()

            for batch_index in range(len(dones)):
                if trajectory_turns[batch_index] == 0:
                    trajectory_turns[batch_index] = step + 2

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        for turns in range(1, self.config.max_turns + 2):
            count = (torch.tensor(trajectory_turns) == turns).sum().item()
            print(f"Finish at the {turns}-th turn: {count}")

        return self._compose_final_output(original_left_side, original_right_side, meta_info), trajectory_turns

    def _compose_final_output(
        self,
        left_side: dict[str, torch.Tensor],
        right_side: dict[str, torch.Tensor],
        meta_info: dict[str, Any],
    ) -> SearchBatch:
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]
        final_output["input_ids"] = torch.cat([left_side["input_ids"], right_side["responses"]], dim=1)
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
                self.tensor_fn.create_attention_mask(final_output["responses_with_info_mask"]),
            ],
            dim=1,
        )
        final_output["position_ids"] = self.tensor_fn.create_position_ids(final_output["attention_mask"])

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
        cur_actions, contents = self.postprocess_predictions(predictions)
        batch_size = len(predictions)
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool)

        search_payload = [
            (
                content,
                problem[index],
                _resolve_ground_truth_text(ground_truth[index]),
            )
            for index, (action, content, active) in enumerate(zip(cur_actions, contents, active_mask.tolist()))
            if active and action == "search"
        ]

        if do_search and search_payload:
            search_results = self.batch_search(search_payload, search_mode, gt_threshold)
        else:
            search_results = [""] * len(search_payload)

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
                next_obs.append(f"\n\n<information>{search_results[search_result_index].strip()}</information>\n\n")
                search_result_index += 1
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                continue

            next_obs.append(
                "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
            )
            dones.append(0)
            valid_action.append(0)
            is_search.append(0)

        return next_obs, dones, valid_action, is_search

    def dynamic_threshold(
        self,
        current_step: int,
        total_steps: int,
        current_turn: int = 1,
        max_turns: int = 5,
    ) -> float:
        del current_turn, max_turns
        if current_step >= total_steps:
            return self.config.end_threshold

        progress = current_step / total_steps
        exp_base = getattr(self.config, "exp_base", 4)
        exp_value = (math.pow(exp_base, progress) - 1) / (exp_base - 1)
        return self.config.start_threshold + exp_value * (
            self.config.end_threshold - self.config.start_threshold
        )

    def postprocess_predictions(self, predictions: list[Any]) -> tuple[list[str | None], list[str]]:
        actions: list[str | None] = []
        contents: list[str] = []
        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            match = ACTION_PATTERN.search(prediction)
            if match:
                actions.append(match.group(1))
                contents.append(match.group(2).strip())
            else:
                actions.append(None)
                contents.append("")
        return actions, contents

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
                except Exception:
                    continue
        return all_search_result

    def _retrieve_from_endpoint(
        self, url: str, query: str, topk: int, retry_attempts: int, sleep_seconds: float = 1.0
    ) -> str:
        import requests  # optional dep; imported here so the class loads without it

        for _ in range(retry_attempts):
            try:
                resp = requests.post(url, json={"queries": [query], "topk": topk}, timeout=10)
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
            f"http://{ip}:6002/retrieve", query, topk,
            self.config.wiki_retry_attempts, self.config.wiki_retry_sleep_seconds,
        )

    def retrieve_from_local(self, query: str, topk: int = 5) -> str:
        """Call the repo's retrieval_server (POST /retrieve) directly."""
        if not self.config.retrieval_url:
            return _NO_INFO
        return self._retrieve_from_endpoint(
            self.config.retrieval_url, query, topk,
            self.config.wiki_retry_attempts, self.config.wiki_retry_sleep_seconds,
        )

    def retrieve_from_google(self, query: str, topk: int, retry_attempt: int | None = None) -> str:
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
                    text_data = f"{item.get('title', '')}{item.get('snippet', '')}".strip()
                    if text_data:
                        search_texts.append(text_data)
                if not search_texts:
                    return _NO_INFO
                return "\n".join([f"Doc {index + 1}: {doc}" for index, doc in enumerate(search_texts)])
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
            doc_texts = self.retrieve_from_wiki(self.config.retriever_ip, query, self.config.topk)
        elif search_mode == "simulate_sft":
            doc_texts = search_simulate_sft(
                self.config.llm_ip,
                self.config.topk,
                self.config.temperature,
                query,
                problem,
                ground_truth,
                gt_threshold,
                llm_max_retries=self.config.llm_max_retries,
            )
        elif search_mode == "simulate_prompt":
            doc_texts = search_simulate_prompt(
                self.config.llm_ip,
                self.config.topk,
                self.config.temperature,
                query,
                problem,
                ground_truth,
                gt_threshold,
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
