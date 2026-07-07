from src.context.models import ContextDocument
from src.internal.servers.web.app import _direct_gate_decision, _levenshtein_lt2, _norm


def test_norm_lowercases_strips_and_collapses():
    assert _norm("  FAISS? ") == "faiss"
    assert _norm("Dense   Retrieval") == "dense retrieval"


def test_levenshtein_lt2_true_for_zero_and_one_edit():
    assert _levenshtein_lt2("faiss", "faiss") is True  # distance 0
    assert _levenshtein_lt2("faiss", "faisz") is True  # 1 substitution
    assert _levenshtein_lt2("faiss", "faisss") is True  # 1 insertion
    assert _levenshtein_lt2("faisss", "faiss") is True  # 1 deletion


def test_levenshtein_lt2_false_for_two_or_more_edits():
    assert _levenshtein_lt2("cat", "dog") is False
    assert _levenshtein_lt2("faiss", "hnsw") is False
    assert _levenshtein_lt2("faiss", "fabss") is True  # exactly 1 sub → still True
    assert _levenshtein_lt2("faiss", "fabsz") is False  # 2 subs


def _d(title, *, score=0.5, content="body", i=1):
    return ContextDocument(
        id=f"D{i}", title=title, content=content, url=None, score=score
    )


def test_exact_title_routes_direct_without_touching_embedder():
    calls = {"n": 0}

    def cos(q, p):
        calls["n"] += 1
        return 0.0

    strong, tier, top, cosine = _direct_gate_decision(
        "FAISS", [_d("faiss"), _d("other", score=0.1, i=2)], cos_min=0.8, cosine_fn=cos
    )
    assert (strong, tier) == (True, "exact")
    assert calls["n"] == 0  # exact match short-circuits before any embed
    assert cosine is None
    assert top == 0.5


def test_fuzzy_typo_with_high_cosine_routes_direct():
    strong, tier, _top, cosine = _direct_gate_decision(
        "faisz", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.95
    )
    assert (strong, tier) == (True, "fuzzy")
    assert cosine == 0.95


def test_fuzzy_near_match_with_low_cosine_escalates():
    # "car" vs "cat": edit distance 1 but unrelated → semantic verify fails.
    strong, tier, _top, _cos = _direct_gate_decision(
        "car", [_d("cat")], cos_min=0.8, cosine_fn=lambda q, p: 0.1
    )
    assert (strong, tier) == (False, "weak")


def test_semantic_equivalent_phrasing_routes_direct():
    strong, tier, _top, cosine = _direct_gate_decision(
        "vector index library", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.9
    )
    assert (strong, tier) == (True, "semantic")
    assert cosine == 0.9


def test_semantic_low_cosine_escalates():
    strong, tier, _top, _cos = _direct_gate_decision(
        "weather today", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: 0.2
    )
    assert (strong, tier) == (False, "weak")


def test_no_embedder_semantic_path_escalates():
    strong, tier, _top, cosine = _direct_gate_decision(
        "vector index", [_d("faiss")], cos_min=0.8, cosine_fn=lambda q, p: None
    )
    assert (strong, tier) == (False, "weak")
    assert cosine is None


def test_empty_docs_escalate():
    strong, tier, top, cosine = _direct_gate_decision(
        "faiss", [], cos_min=0.8, cosine_fn=lambda q, p: 0.9
    )
    assert (strong, tier, top, cosine) == (False, "weak", 0.0, None)
