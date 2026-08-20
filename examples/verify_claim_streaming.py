"""Check that Assist really streams verified claims, against a live stack.

Claim-level streaming rests on one invariant: once a claim is emitted, the final
answer must equal the join of the emitted claims, in emission order. Unit tests
pin the pieces, but they cannot prove the thing the design actually depends on —
that the *provider* emits ``abstain`` before ``claims``. If it does not, the
incremental reader gives up on every draft, no claim ever streams, and the
feature is silently inert behind a green suite. That failure is invisible except
from end to end, which is what this script is for.

There is no separate ordering probe here on purpose: the reader only streams when
it sees ``abstain`` first, so claims arriving at all *is* the proof it held.

Run it with the stack up (retrieval on :8001, web backend on :7860) and an
``OPENAI_API_KEY`` in the environment, or Assist never reaches the grounded path:

    python -m examples.verify_claim_streaming --query "What is FAISS?"

Exit codes: 0 the stream satisfied the invariant, 1 it did not, 2 the stream
could not be read at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import requests

DEFAULT_URL = "http://127.0.0.1:7860/api/agent/stream"
DEFAULT_QUERY = "What is FAISS?"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 300

TimedEvent = tuple[float, dict]


@dataclass(frozen=True)
class StreamSummary:
    """What one Assist stream did, and every way it failed the invariant."""

    claims: tuple[str, ...]
    answer: str | None
    time_to_first_claim: float | None
    time_to_answer: float | None
    diagnostics: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def lead(self) -> float | None:
        """How far ahead of completion the first claim arrived, in seconds."""
        if self.time_to_first_claim is None or self.time_to_answer is None:
            return None
        return self.time_to_answer - self.time_to_first_claim


def summarize(events: Sequence[TimedEvent]) -> StreamSummary:
    """Reduce a timed SSE event sequence to the claim-streaming verdict."""
    claim_events = [
        (at, event["text"]) for at, event in events if event.get("type") == "claim"
    ]
    answers = [
        (at, event.get("text", ""))
        for at, event in events
        if event.get("type") == "answer"
    ]
    claims = tuple(text for _, text in claim_events)
    answer = answers[0][1] if answers else None
    time_to_first_claim = claim_events[0][0] if claim_events else None
    time_to_answer = answers[0][0] if answers else None

    diagnostics: list[str] = []
    for _, event in events:
        if event.get("type") == "error":
            diagnostics.append(
                f"the server emitted an error event: {event.get('detail')}"
            )

    if answer is None:
        diagnostics.append(
            "no answer event arrived, so there is nothing to check the claims against"
        )
    if not claims:
        diagnostics.append(
            "no claim streamed — the reader gives up unless the provider emits "
            "abstain before claims, so the feature is inert on this provider"
        )
    elif time_to_answer is not None:
        late = [at for at, _ in claim_events if at >= time_to_answer]
        if late:
            diagnostics.append(
                f"{len(late)} claim event(s) arrived at or after the answer event — "
                "a claim has to arrive before it, or streaming buys no lead"
            )

    if claims and answer is not None and " ".join(claims) != answer:
        diagnostics.append(
            "the answer is not the join of the streamed claims — a claim was "
            "shown that the final answer does not contain, or vice versa"
        )

    return StreamSummary(
        claims=claims,
        answer=answer,
        time_to_first_claim=time_to_first_claim,
        time_to_answer=time_to_answer,
        diagnostics=tuple(diagnostics),
    )


def iter_stream(
    *, url: str, query: str, top_k: int, timeout: int
) -> Iterator[TimedEvent]:
    """POST to the SSE endpoint, yielding each event with its arrival time."""
    started = time.monotonic()
    response = requests.post(
        url, json={"query": query, "top_k": top_k}, stream=True, timeout=timeout
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield time.monotonic() - started, json.loads(line[len("data: ") :])


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def render_report(summary: StreamSummary) -> str:
    lines = [
        f"claims streamed     : {len(summary.claims)}",
        f"time-to-first-claim : {_seconds(summary.time_to_first_claim)}",
        f"time-to-answer      : {_seconds(summary.time_to_answer)}",
        f"lead                : {_seconds(summary.lead)}",
    ]
    for index, claim in enumerate(summary.claims, start=1):
        lines.append(f"  claim {index}: {claim}")
    lines.extend(f"DIAGNOSTIC: {diagnostic}" for diagnostic in summary.diagnostics)
    lines.append(
        "PASS: the answer is the join of the claims streamed before it"
        if summary.ok
        else "FAIL"
    )
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    open_stream: object = iter_stream,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    try:
        events: Iterable[TimedEvent] = open_stream(  # type: ignore[operator]
            url=args.url, query=args.query, top_k=args.top_k, timeout=args.timeout
        )
        timed_events = list(events)
    except (OSError, ValueError) as exc:
        print(f"could not read the stream at {args.url}: {exc}", file=sys.stderr)
        return 2

    summary = summarize(timed_events)
    print(f"query: {args.query}")
    print(render_report(summary))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
