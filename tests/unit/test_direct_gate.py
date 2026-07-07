from src.internal.servers.web.app import _levenshtein_lt2, _norm


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
