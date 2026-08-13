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


# Only 3 clean (bulk-) queries exist in this fixture, so slice_size must stay
# well under that. At 1, the split is deterministic regardless of seed: the
# lone "search"-route clean query is the only member of its pool, so
# rng.sample of a 1-item list always returns that same item.
_SLICE_SIZE = 1


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
        model_name="test-encoder",
        slice_size=_SLICE_SIZE,
    )


def test_split_puts_every_legacy_query_in_tuning_and_reports_split_sizes(
    tmp_path, monkeypatch
):
    """The legacy queries are contaminated, so all of them are free to tune on.

    With ``_SLICE_SIZE`` clean queries drawn into tuning, tuning holds the 2
    legacy queries plus that slice, and the rest of the clean set becomes the
    untouched test slice.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["split"]["tuning_size"] == 2 + _SLICE_SIZE
    assert report["split"]["test_size"] == 3 - _SLICE_SIZE
    assert report["legacy_30"]["total_queries"] == 2
    assert report["tuning"]["total_queries"] == report["split"]["tuning_size"]
    assert report["test_slice"]["total_queries"] == report["split"]["test_size"]
    assert report["bulk"]["total_queries"] == len(_BULK_QUERIES)
    # Pooled, not just parallel: nothing dropped, nothing double-counted.
    assert (
        report["tuning"]["total_queries"] + report["test_slice"]["total_queries"]
        == report["bulk"]["total_queries"]
    )


def test_accuracy_is_argmax_over_every_query_not_only_those_above_a_threshold(
    tmp_path, monkeypatch
):
    """A confidently-wrong, low-similarity prediction must still count.

    If accuracy were computed only over records clearing some confidence
    threshold, the low-confidence miss ("clean low confidence miss", best
    route score only 0.3) is exactly the record a bug would drop. With
    ``_SLICE_SIZE`` == 1 the split is deterministic: the lone search-route
    clean query ("clean search query") is the only one that can join tuning,
    so the test slice always holds the other two -- the correct tool query
    and the mislabeled low-confidence miss.
    """
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["legacy_30"]["accuracy"] == pytest.approx(1.0)
    assert report["tuning"]["accuracy"] == pytest.approx(1.0)
    assert report["test_slice"]["accuracy"] == pytest.approx(1 / 2)
    assert report["bulk"]["accuracy"] == pytest.approx(4 / 5)


def test_out_of_scope_uses_separability_on_the_test_slice(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True)

    # test-slice confidences: "clean tool query" 0.85, "clean low confidence
    # miss" 0.3 (its best route score, regardless of correctness).
    mean_in_scope = (0.85 + 0.3) / 2
    # probe confidences: "probe one" best route score 0.1, "probe two" 0.2.
    mean_out_of_scope = (0.1 + 0.2) / 2

    out_of_scope = report["out_of_scope"]
    assert out_of_scope["counts"] == {"in_scope": 2, "out_of_scope": 2}
    assert out_of_scope["min_in_scope"] == pytest.approx(0.3)
    assert out_of_scope["max_out_of_scope"] == pytest.approx(0.2)
    assert out_of_scope["raw_margin"] == pytest.approx(
        mean_in_scope - mean_out_of_scope
    )
    # Alias kept for the raw-cosine-units bar in the pinned section below.
    assert out_of_scope["separation_margin"] == pytest.approx(
        out_of_scope["raw_margin"]
    )
    # Perfectly separated on this fixture: both in-scope scores clear both
    # out-of-scope scores.
    assert out_of_scope["auc"] == pytest.approx(1.0)


def test_run_index_evaluation_raises_on_a_stale_canonical_fingerprint(
    tmp_path, monkeypatch
):
    """Editing the canonical file without rebuilding must fail loudly.

    Otherwise `evaluate` silently scores the index built from the *old*
    canonical file and stamps the report with the new canonical path, so an
    operator reads the old accuracy as if it measured their edit.
    """
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)
    canonical_path = tmp_path / "canonical.json"
    edited = json.loads(canonical_path.read_text(encoding="utf-8"))
    edited[0]["text"] = "canonical search 0 but edited after the index was built"
    canonical_path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=canonical_path,
            output_path=tmp_path / "report.json",
            model_name="test-encoder",
            slice_size=_SLICE_SIZE,
        )


def test_run_index_evaluation_passes_when_the_canonical_fingerprint_matches(
    tmp_path, monkeypatch
):
    report = _run(tmp_path, monkeypatch=monkeypatch)

    assert report["index"]["fingerprint"]


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
            model_name="test-encoder",
            slice_size=1,
        )


def test_run_index_evaluation_raises_when_the_encoder_does_not_match_the_index(
    tmp_path, monkeypatch
):
    """Both all-MiniLM-L6-v2 and e5-small-v2 are 384-dimensional, so nothing
    else catches a mismatched encoder: no shape error, no exception, just a
    confident, meaningless number -- exactly what happened once already,
    scoring a stale MiniLM-built index against e5-encoded queries.

    The index built here is structurally valid (right dimensionality, right
    row count); its only defect is that it declares a different encoder than
    the one this evaluation asks for. ``_encode_stub`` ignores ``model_name``
    entirely, so without the guard this would run to completion and score
    without error -- the guard is the only thing standing between a real
    mismatch and a silently wrong report.
    """
    monkeypatch.setattr(intent_index_eval, "encode_texts", _encode_stub)
    index_dir = _build_index(tmp_path)  # built with model_name="test-encoder"

    with pytest.raises(ValueError) as exc_info:
        intent_index_eval.run_index_evaluation(
            index_path=index_dir,
            eval_queries_path=_eval_queries_path(tmp_path, _BULK_QUERIES),
            hard_queries_path=None,
            out_of_scope_path=None,
            canonical_path=tmp_path / "canonical.json",
            output_path=tmp_path / "report.json",
            model_name="different-encoder",
            slice_size=_SLICE_SIZE,
        )

    message = str(exc_info.value)
    assert "test-encoder" in message
    assert "different-encoder" in message


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


def test_leave_one_out_default_top_k_matches_the_module_constant():
    """Nothing passing top_k must behave exactly as before the parameter existed."""
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

    default = intent_index_eval.leave_one_out_route_accuracy(index)
    explicit = intent_index_eval.leave_one_out_route_accuracy(
        index, top_k=intent_index_eval.TOP_K
    )

    assert default == explicit


def test_top_k_sweep_reports_every_configured_k_without_changing_the_shipped_report(
    tmp_path, monkeypatch
):
    """The sweep is evidence, not a selection: the shipped headline is untouched."""
    report = _run(tmp_path, monkeypatch=monkeypatch, out_of_scope=True, hard=None)

    sweep = report["top_k_sweep"]
    assert [row["top_k"] for row in sweep["rows"]] == list(
        intent_index_eval._SWEEP_TOP_K
    )
    for row in sweep["rows"]:
        assert 0.0 <= row["test_slice_accuracy"] <= 1.0
        assert 0.0 <= row["leave_one_out_accuracy"] <= 1.0
        assert "separation_margin" in row
        assert "hard_accuracy" not in row  # no hard queries were supplied

    # The row at the shipped TOP_K reproduces the report's own headline numbers.
    shipped = next(
        row for row in sweep["rows"] if row["top_k"] == intent_index_eval.TOP_K
    )
    assert shipped["test_slice_accuracy"] == pytest.approx(
        report["test_slice"]["accuracy"]
    )
    assert shipped["leave_one_out_accuracy"] == pytest.approx(
        report["leave_one_out"]["accuracy"]
    )
    assert shipped["separation_margin"] == pytest.approx(
        report["out_of_scope"]["separation_margin"]
    )


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
    from src.model.intent_encoder import DEFAULT_ENCODER
    from src.model.intent_index_eval import run_index_evaluation
    from src.model.intent_knn import INDEX_FILENAME, IntentIndex

    index = DATA / "intent_index"
    if not (index / INDEX_FILENAME).exists():
        pytest.skip(
            "run `python -m src.model.intent_index_cli build --canonical "
            f"data/intent_canonical.json --output {index}` to measure the bars"
        )
    on_disk_encoder = IntentIndex.load(index / INDEX_FILENAME).encoder
    if on_disk_encoder != DEFAULT_ENCODER:
        # run_index_evaluation now rejects this itself (an index whose
        # encoder differs from the one in use scores silent garbage
        # otherwise), so a stale on-disk index is a skip here rather than an
        # error: these bars are unmeasurable until the index is rebuilt.
        pytest.skip(
            f"data/intent_index was built with encoder {on_disk_encoder!r}, "
            f"but the current encoder is {DEFAULT_ENCODER!r}; run "
            "`python -m src.model.intent_index_cli build --canonical "
            f"data/intent_canonical.json --output {index}` to re-measure "
            "the bars"
        )
    return run_index_evaluation(
        index_path=index,
        eval_queries_path=DATA / "intent_eval_queries.json",
        hard_queries_path=DATA / "intent_eval_hard.json",
        out_of_scope_path=DATA / "intent_out_of_scope.json",
        canonical_path=DATA / "intent_canonical.json",
        output_path=index / "evaluation_report.json",
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "_BULK_ACCURACY_FLOOR=0.63 is a raw-cosine-units bar calibrated for "
        "all-MiniLM-L6-v2 and is encoder-specific; it does not survive the "
        "e5-small-v2 swap (PR feat/intent-encoder-e5 Task 1) — the on-disk "
        "index this reads was built with the previous encoder. Re-derived "
        "and re-pinned once Task 3 of "
        "docs/superpowers/plans/2026-08-13-intent-encoder-e5.md rebuilds the "
        "index and re-measures."
    ),
)
def test_index_holds_the_bulk_accuracy_bar():
    assert _report()["bulk"]["accuracy"] >= _BULK_ACCURACY_FLOOR


@pytest.mark.xfail(
    strict=False,
    reason=(
        "_SEPARATION_MARGIN_FLOOR=0.10 is a raw-cosine-units bar calibrated "
        "for all-MiniLM-L6-v2 and is encoder-specific; it does not survive "
        "the e5-small-v2 swap (PR feat/intent-encoder-e5 Task 1) — the "
        "on-disk index this reads was built with the previous encoder. "
        "Re-derived and re-pinned once Task 3 of "
        "docs/superpowers/plans/2026-08-13-intent-encoder-e5.md rebuilds the "
        "index and re-measures."
    ),
)
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
