"""Distill the current intent router (regex + LLM) into the trained classifier.

Label a query set with the router's own decisions, then train the MLP on those
labels so it learns intent cues (verb/structure) instead of topic tokens.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.model.intent_classifier import IntentPipeline
from src.model.intent_training import train_intent_classifier


def label_query(query: str, *, llm=None) -> tuple[str, str]:
    """Label one query with the router. Returns (label, teacher)."""
    from src.internal.servers.web.intent_routing import (
        _regex_route,
        _rule_based_route,
        classify_route,
    )

    regex = _regex_route(query)
    if regex is not None:
        return regex.value, "regex"
    if llm is not None:
        try:
            return classify_route(query, llm)[0].value, "llm"
        except Exception:
            pass
    return _rule_based_route(query).value, "rule_based"


def build_distillation_examples(queries, *, llm=None) -> list[dict]:
    """Map queries to [{text, label}], dropping blank queries."""
    examples: list[dict] = []
    for q in queries:
        text = (q or "").strip()
        if not text:
            continue
        label, _teacher = label_query(text, llm=llm)
        examples.append({"text": text, "label": label})
    return examples


@dataclass(frozen=True)
class DistillResult:
    pipeline: IntentPipeline
    num_examples: int
    label_counts: dict[str, int]
    teacher_counts: dict[str, int]


def distill_and_train(
    queries,
    *,
    output_path: Path,
    examples_path: Path,
    llm=None,
    epochs: int = 15,
    lr: float = 1e-3,
    min_freq: int = 1,
) -> DistillResult:
    """Label queries with the router, write examples, and train the classifier."""
    examples: list[dict] = []
    teacher_counts: Counter = Counter()
    label_counts: Counter = Counter()
    for q in queries:
        text = (q or "").strip()
        if not text:
            continue
        label, teacher = label_query(text, llm=llm)
        examples.append({"text": text, "label": label})
        teacher_counts[teacher] += 1
        label_counts[label] += 1
    if not examples:
        raise ValueError("No non-empty queries to distill.")

    examples_path = Path(examples_path)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2))

    result = train_intent_classifier(
        examples_path=examples_path,
        output_path=Path(output_path),
        epochs=epochs,
        lr=lr,
        min_freq=min_freq,
    )
    return DistillResult(
        pipeline=result.pipeline,
        num_examples=result.num_examples,
        label_counts=dict(label_counts),
        teacher_counts=dict(teacher_counts),
    )


def load_queries_from_file(path) -> list[str]:
    """Load queries from a .txt (one per line) or a .json list of str/dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path!r}")
    if path.suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Query file is not valid JSON: {path!r}") from exc
        out: list[str] = []
        for item in raw:
            text = (
                item
                if isinstance(item, str)
                else (item.get("text") or item.get("question") or "")
            )
            if text and text.strip():
                out.append(text.strip())
        return out
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_teacher_llm(args):
    if not args.vllm_url:
        return None
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=args.model_provider,
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.vllm_url,
        )
    )


def _collect_queries(args) -> list[str]:
    queries: list[str] = []
    if args.queries_file:
        queries.extend(load_queries_from_file(args.queries_file))
    if args.from_db:
        from src.internal.db.store import AgenticSearchStore

        store = AgenticSearchStore(args.from_db)
        queries.extend(store.get_user_query_texts())
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Distill the intent router into a trained classifier."
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="File of queries (.txt one-per-line or .json list).",
    )
    parser.add_argument(
        "--from-db",
        type=str,
        default=None,
        help="SQLite store path to pull distinct logged user queries from.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .pt path.")
    parser.add_argument(
        "--examples-out",
        type=str,
        default=None,
        help="Where to write the labeled examples JSON (defaults beside --output).",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for the LLM teacher (ambiguous tail). Omit for offline regex+rule-based.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Teacher model name (with --vllm-url)."
    )
    parser.add_argument(
        "--model-provider", dest="model_provider", type=str, default="openai"
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.vllm_url and not args.model:
        parser.error("--model is required when --vllm-url is given")

    queries = _collect_queries(args)
    if not queries:
        parser.error("no queries collected; provide --queries-file and/or --from-db")

    output = Path(args.output)
    examples_out = (
        Path(args.examples_out)
        if args.examples_out
        else output.with_suffix(".examples.json")
    )
    llm = _build_teacher_llm(args)

    result = distill_and_train(
        queries,
        output_path=output,
        examples_path=examples_out,
        llm=llm,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"queries={len(queries)} num_examples={result.num_examples}")
    print(f"label_counts={result.label_counts}")
    print(f"teacher_counts={result.teacher_counts}")
    print(f"saved -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
