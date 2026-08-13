import functools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from src.model import intent_index_cli, intent_index_eval
from src.model.intent_knn import CanonicalExample, IntentIndex

# Orthogonal one-hot vectors in (search, chat, tool) component order, so
# top-3-mean cosine similarity is exactly 0.0 or 1.0 for a clean match and any
# fractional value we choose for a deliberately ambiguous query.
_SEARCH = [1.0, 0.0, 0.0]
_CHAT = [0.0, 1.0, 0.0]
_TOOL = [0.0, 0.0, 1.0]

_MODULE_FOR_ROUTE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}

_CANONICAL = [
    {
        "id": f"c-{route}-{i}",
        "text": f"canonical {route} {i}",
        "route": route,
        "modules": [_MODULE_FOR_ROUTE[route]],
    }
    for route in ("search", "chat", "tool")
    for i in range(2)
]

_VECTORS_BY_TEXT = {
    **{
        record["text"]: vector
        for record, vector in zip(
            _CANONICAL, [_SEARCH, _SEARCH, _CHAT, _CHAT, _TOOL, _TOOL]
        )
    },
    # 0.85 on-axis, not 1.0: a clear, correct match without tripping the
    # >=0.95 leakage guard the way an exact-axis vector would.
    "legacy search query": [0.85, 0.05, 0.05],
    "legacy chat query": [0.05, 0.85, 0.05],
    "clean search query": [0.85, 0.05, 0.05],
    "clean tool query": [0.05, 0.05, 0.85],
    # Deliberately mislabeled ("tool") and low confidence (best route scores
    # only 0.3): the worst case for a threshold-gated accuracy computation,
    # since a buggy filter would most plausibly drop exactly this kind of
    # record and inflate accuracy by omission.
    "clean low confidence miss": [0.3, 0.25, 0.2],
    "probe one": [0.1, 0.05, 0.02],
    "probe two": [0.2, 0.0, 0.0],
}


def _encode_stub(texts, *, model_name="test-encoder"):
    return np.array([_VECTORS_BY_TEXT[text] for text in texts], dtype=np.float32)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_index(tmp_path: Path) -> Path:
    output = tmp_path / "index"
    intent_index_cli.build_index(
        _write_json(tmp_path / "canonical.json", _CANONICAL),
        output,
        model_name="test-encoder",
        encode=_encode_stub,
    )
    return output


def _eval_queries_path(tmp_path: Path, records) -> Path:
    return _write_json(tmp_path / "eval_queries.json", records)


_BULK_QUERIES = [
    {
        "id": "eval-a",
        "text": "legacy search query",
        "label": "search",
        "modules": ["lookup_fact"],
    },
    {
        "id": "eval-b",
        "text": "legacy chat query",
        "label": "chat",
        "modules": ["explain"],
    },
    {
        "id": "bulk-a",
        "text": "clean search query",
        "label": "search",
        "modules": ["lookup_fact"],
    },
    {
        "id": "bulk-b",
        "text": "clean tool query",
        "label": "tool",
        "modules": ["schedule"],
    },
    {
        "id": "bulk-c",
        "text": "clean low confidence miss",
        "label": "tool",
        "modules": ["schedule"],
    },
]

_PROBES = [
    {"id": "oos-1", "text": "probe one"},
    {"id": "oos-2", "text": "probe two"},
]


def _run(tmp_path, *, monkeypatch, out_of_scope=None, hard=None):
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    return intent_index_eval.run_index_evaluation(
        index_path=index_dir,
        eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
        hard_queries_path=hard,
        out_of_scope_path=(
            _write_json(tmp_path / "oos.json", _PROBES) if out_of_scope else None
        ),
        canonical_path=tmp_path / "canonical.json",
        output_path=tmp_path / "report.json",
    )


def test_slice_split_puts_eval_prefixed_in_legacy_and_bulk_prefixed_in_clean(
    tmp_path, monkeypatch
):
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["legacy_30"]["total_queries"] == 2
    assert report["clean_151"]["total_queries"] == 3
    assert report["bulk"]["total_queries"] == len(_BULK_QUERIES)
    # Pooled, not just parallel: nothing dropped, nothing double-counted.
    assert (
        report["legacy_30"]["total_queries"] + report["clean_151"]["total_queries"]
        == report["bulk"]["total_queries"]
    )


def test_accuracy_is_argmax_over_every_query_not_only_those_above_a_threshold(
    tmp_path, monkeypatch
):
    """A confidently-wrong, low-similarity prediction must still count.

    If accuracy were computed only over records clearing some confidence
    threshold, the low-confidence miss ("clean low confidence miss", best
    route score only 0.3) is exactly the record a bug would drop, and both
    clean_151 and bulk accuracy would read 1.0 instead of the true values.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["legacy_30"]["accuracy"] == pytest.approx(1.0)
    assert report["clean_151"]["accuracy"] == pytest.approx(2 / 3)
    assert report["bulk"]["accuracy"] == pytest.approx(4 / 5)


def test_out_of_scope_margin_is_mean_in_scope_minus_mean_out_of_scope_on_clean(
    tmp_path, monkeypatch
):
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)

    # clean_151 confidences: search query 0.85, tool query 0.85, low-
    # confidence miss 0.3 (its best route score, regardless of correctness).
    mean_in_scope = (0.85 + 0.85 + 0.3) / 3
    # probe confidences: "probe one" best route score 0.1, "probe two" 0.2.
    mean_out_of_scope = (0.1 + 0.2) / 2

    out_of_scope = report["out_of_scope"]
    assert out_of_scope["mean_in_scope_confidence"] == pytest.approx(mean_in_scope)
    assert out_of_scope["mean_out_of_scope_confidence"] == pytest.approx(
        mean_out_of_scope
    )
    assert out_of_scope["separation_margin"] == pytest.approx(
        mean_in_scope - mean_out_of_scope
    )


def test_run_index_evaluation_raises_on_leakage_instead_of_scoring_anyway(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    leaking_queries = [
        {
            "id": "bulk-leak",
            "text": "canonical search 0",
            "label": "search",
            "modules": ["lookup_fact"],
        },
    ]

    with pytest.raises(ValueError, match="leak"):
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, leaking_queries),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=tmp_path / "canonical.json",
            output_path=tmp_path / "report.json",
        )


def test_leave_one_out_excludes_each_example_from_its_own_scoring():
    """A singleton route's only example cannot vote for itself once excluded.

    "tool" has exactly one canonical example. Scored against the full index
    including itself (the "control" a self-inclusive scorer would produce),
    it always wins its own route with similarity 1.0. Excluded from its own
    route's candidate pool (real leave-one-out), that route has no
    remaining examples at all, so the example is misclassified. The control
    must therefore score strictly higher than genuine leave-one-out on this
    fixture.
    """
    examples = [
        CanonicalExample("s0", "search zero", "search", ("lookup_fact",)),
        CanonicalExample("s1", "search one", "search", ("lookup_fact",)),
        CanonicalExample("c0", "chat zero", "chat", ("explain",)),
        CanonicalExample("c1", "chat one", "chat", ("explain",)),
        CanonicalExample("t0", "tool zero", "tool", ("schedule",)),
    ]
    vectors = np.array([_SEARCH, _SEARCH, _CHAT, _CHAT, _TOOL], dtype=np.float32)
    index = IntentIndex(
        examples=examples, vectors=vectors, encoder="test", fingerprint="test"
    )

    control_correct = sum(
        index.decide(
            vectors[i], min_confidence=0.0, min_margin=0.0, min_module_score=0.0
        ).route
        == examples[i].route
        for i in range(len(examples))
    )
    control_accuracy = control_correct / len(examples)

    loo = intent_index_eval.leave_one_out_route_accuracy(index)

    assert control_accuracy == pytest.approx(1.0)
    assert loo["n"] == len(examples)
    assert loo["accuracy"] < control_accuracy


# ---------------------------------------------------------------------------
# The pinned bars.
#
# Everything above runs on a synthetic index and no encoder. Everything below
# measures the *committed* canonical set with the real MiniLM encoder, so it
# skips wherever sentence-transformers or the built index is absent.
#
# Raise a floor when a run beats it; never lower one without recording why in
# the commit message.
# ---------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[2] / "data"

# Measured 2026-08-13 on the 280-example canonical set (task-8 report), pinned
# ~0.02 below the measurement so ordinary encoder/float drift cannot trip them:
#   bulk_181 route accuracy   0.6519  -> floor 0.63
#   out-of-scope margin       0.1188  -> floor 0.10
#   p95 routing latency       5.88 ms -> ceiling 25.0 ms
_BULK_ACCURACY_FLOOR = 0.63
_SEPARATION_MARGIN_FLOOR = 0.10
_P95_LATENCY_CEILING_MS = 25.0


@functools.lru_cache(maxsize=1)
def _report():
    pytest.importorskip("sentence_transformers")
    from src.model.intent_index_eval import run_index_evaluation

    index = DATA / "intent_index"
    if not (index / "index.npz").exists():
        pytest.skip(
            "run `python -m src.model.intent_index_cli build --canonical "
            f"data/intent_canonical.json --output {index}` to measure the bars"
        )
    return run_index_evaluation(
        index_path=index,
        eval_queries_path=DATA / "intent_eval_queries.json",
        hard_queries_path=DATA / "intent_eval_hard.json",
        out_of_scope_path=DATA / "intent_out_of_scope.json",
        canonical_path=DATA / "intent_canonical.json",
        output_path=index / "evaluation_report.json",
    )


def test_index_holds_the_bulk_accuracy_bar():
    assert _report()["bulk"]["accuracy"] >= _BULK_ACCURACY_FLOOR


def test_out_of_scope_requests_score_below_in_scope_requests():
    assert _report()["out_of_scope"]["separation_margin"] >= _SEPARATION_MARGIN_FLOOR


def test_every_module_has_enough_canonical_support_to_be_emitted():
    assert _report()["index"]["low_support_modules"] == []


def test_the_report_covers_the_whole_bulk_set():
    """A silently shrunken eval set would inflate every later number."""
    pytest.importorskip("sentence_transformers")
    assert _report()["bulk"]["total_queries"] >= 170
    assert _report()["legacy_30"]["total_queries"] == 30


def test_routing_one_request_stays_under_the_latency_ceiling():
    """Encode plus decide, the whole serving cost of a route decision."""
    pytest.importorskip("sentence_transformers")
    _report()  # skips for the same reasons as the bars above

    from src.model.intent_encoder import encode_texts
    from src.model.intent_knn import INDEX_FILENAME, IntentIndex

    index = IntentIndex.load(DATA / "intent_index" / INDEX_FILENAME)
    query = "book the meeting room for tomorrow afternoon"
    decide = functools.partial(
        index.decide, min_confidence=0.30, min_margin=0.02, min_module_score=0.45
    )
    for _ in range(5):
        decide(encode_texts([query])[0])

    timings = []
    for _ in range(50):
        start = perf_counter()
        decide(encode_texts([query])[0])
        timings.append((perf_counter() - start) * 1_000)

    p95 = sorted(timings)[int(0.95 * (len(timings) - 1))]
    assert p95 <= _P95_LATENCY_CEILING_MS, p95
