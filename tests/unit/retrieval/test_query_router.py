from __future__ import annotations

from src.internal.retrieval.query_router import QueryRouter


def test_heuristic_short_keyword_query():
    cfg = QueryRouter(model_path=None).predict("faiss index")
    assert cfg.keywords is True
    assert cfg.decompose is False


def test_heuristic_multi_clause_query():
    cfg = QueryRouter(model_path=None).predict(
        "Compare dense and sparse retrieval and explain when each wins"
    )
    assert cfg.decompose is True


def test_heuristic_date_query_constructs_filters():
    cfg = QueryRouter(model_path=None).predict("FAISS papers after 2023")
    assert cfg.construct_filters is True


def test_from_env_disabled(monkeypatch):
    monkeypatch.delenv("QT_ROUTER", raising=False)
    assert QueryRouter.from_env() is None


def test_trained_artifact_round_trips(tmp_path):
    from src.training.train_query_router import train

    path = str(tmp_path / "router.joblib")
    train(path)
    cfg = QueryRouter(model_path=path).predict("faiss index")
    # A loaded model returns a valid config (booleans), not a crash.
    assert isinstance(cfg.decompose, bool)
