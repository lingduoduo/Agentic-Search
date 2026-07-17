from __future__ import annotations


from src.training.search_environment import SearchEnvironment


def _env(**kw):
    kw.setdefault("seed", 0)
    return SearchEnvironment(**kw)


def test_reset_selects_a_known_question_and_clears_state():
    env = _env()
    env.reset()
    qids = {q["id"] for q in env.questions}
    assert env.current_question in qids
    assert env.gathered == set()
    assert env.steps == 0
    assert env.game_over is False
    assert env.victory is False


def test_reset_round_robins_over_questions():
    env = _env()
    seen = []
    for _ in range(len(env.questions) + 1):
        env.reset()
        seen.append(env.current_question)
    # First question repeats after a full cycle.
    assert seen[0] == seen[len(env.questions)]
    assert len(set(seen[: len(env.questions)])) == len(env.questions)


def test_available_actions_cover_all_topics_plus_answer_stop():
    env = _env()
    env.reset()
    actions = env.get_available_actions()
    for topic in env.topics:
        assert f"retrieve:{topic}" in actions
    assert "answer" in actions
    assert "stop" in actions
