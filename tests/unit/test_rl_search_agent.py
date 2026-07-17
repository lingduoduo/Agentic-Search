from __future__ import annotations

import random

import numpy as np
import pytest

from src.training.rl_agent import QLearningAgent
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


def _reset_to(env, question_id):
    """Reset until the given question is active (round-robin makes this deterministic)."""
    for _ in range(len(env.questions)):
        env.reset()
        if env.current_question == question_id:
            return
    raise AssertionError(f"question {question_id} not found")


def test_retrieve_relevant_new_fact_nets_positive_and_gathers():
    env = _env()
    _reset_to(env, "what_is_faiss")  # requires {"faiss"}
    feedback, reward, done = env.execute_action("retrieve:faiss")
    assert "faiss" in env.gathered
    assert reward == 1  # +2 relevance, -1 step cost
    assert done is False


def test_retrieve_irrelevant_costs_one_and_gathers_nothing():
    env = _env()
    _reset_to(env, "what_is_faiss")
    feedback, reward, done = env.execute_action("retrieve:bm25")
    assert env.gathered == set()
    assert reward == -1
    assert done is False


def test_retrieve_duplicate_costs_one():
    env = _env()
    _reset_to(env, "what_is_faiss")
    env.execute_action("retrieve:faiss")
    _, reward, _ = env.execute_action("retrieve:faiss")
    assert reward == -1


def test_answer_with_all_facts_wins():
    env = _env()
    _reset_to(env, "what_is_faiss")
    env.execute_action("retrieve:faiss")
    feedback, reward, done = env.execute_action("answer")
    assert reward == 50
    assert env.victory is True
    assert done is True
    assert env.game_over is True


def test_answer_early_penalized_and_terminal():
    env = _env()
    _reset_to(env, "dense_vs_sparse")  # requires {"faiss", "bm25"}
    env.execute_action("retrieve:faiss")
    feedback, reward, done = env.execute_action("answer")
    assert reward == -10
    assert env.victory is False
    assert done is True


def test_stop_penalized_and_terminal():
    env = _env()
    _reset_to(env, "what_is_faiss")
    _, reward, done = env.execute_action("stop")
    assert reward == -5
    assert done is True
    assert env.game_over is True


def test_step_budget_terminates_with_penalty():
    env = _env(max_steps=3)
    _reset_to(env, "hybrid_pipeline")  # requires 3 facts, unreachable in 3 wasted steps
    rewards = []
    done = False
    for _ in range(3):
        _, reward, done = env.execute_action("retrieve:faiss")  # irrelevant here
        rewards.append(reward)
    assert done is True
    assert rewards[-1] == -10  # budget-exceeded penalty replaces the step reward
    assert env.game_over is True


def test_stochastic_relevant_retrieval_can_fail():
    env = _env(stochastic=True, fail_prob=1.0, seed=0)
    _reset_to(env, "what_is_faiss")
    _, reward, _ = env.execute_action("retrieve:faiss")
    assert env.gathered == set()  # forced failure
    assert reward == -1


def test_dqn_agent_is_gone():
    import src.training.rl_agent as mod

    assert not hasattr(mod, "DQNAgent")


def test_state_hash_is_stable_and_reflects_gathered():
    env = _env()
    _reset_to(env, "what_is_faiss")
    agent = QLearningAgent()
    h0 = agent._get_state_hash(env)
    assert agent._get_state_hash(env) == h0  # deterministic
    env.execute_action("retrieve:faiss")
    assert agent._get_state_hash(env) != h0  # gathering changes the state


def test_update_q_value_matches_bellman():
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
    agent.q_table["s'"]["a1"] = 10.0
    agent.q_table["s'"]["a2"] = 4.0
    # current Q(s,a) defaults to 0.0
    agent.update_q_value(
        "s", "a", reward=2.0, next_state="s'", next_actions=["a1", "a2"], done=False
    )
    # target = 2 + 0.9*10 = 11 ; Q <- 0 + 0.5*(11-0) = 5.5
    assert agent.q_table["s"]["a"] == pytest.approx(5.5)


def test_update_q_value_terminal_uses_reward_only():
    agent = QLearningAgent(learning_rate=1.0, discount_factor=0.9)
    agent.update_q_value(
        "s", "a", reward=7.0, next_state="s'", next_actions=[], done=True
    )
    assert agent.q_table["s"]["a"] == pytest.approx(7.0)


def test_epsilon_decays_toward_min():
    random.seed(0)
    np.random.seed(0)
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1)
    env = SearchEnvironment(seed=0)
    start = agent.epsilon
    for _ in range(50):
        agent.train_episode(env)
    assert agent.epsilon < start
    assert agent.epsilon >= agent.epsilon_min


def test_agent_learns_single_fact_question():
    random.seed(0)
    np.random.seed(0)
    # Environment with only the trivial one-fact question, so learning is fast.
    env = SearchEnvironment(seed=0)
    env.questions = [
        {"id": "what_is_faiss", "required_facts": {"faiss"}, "answer": "faiss"}
    ]
    env._question_idx = 0
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05)
    for _ in range(400):
        agent.train_episode(env)
    wins = 0
    agent.epsilon = 0.0
    for _ in range(50):
        env.reset()
        while not env.game_over:
            action = agent.choose_action(env, training=False)
            env.execute_action(action)
        wins += int(env.victory)
    assert wins / 50 > 0.9


def test_save_and_load_round_trips(tmp_path):
    agent = QLearningAgent()
    agent.q_table["s"]["a"] = 3.14
    agent.total_episodes = 5
    agent.victories = 2
    path = tmp_path / "qtable.pkl"
    agent.save(str(path))

    fresh = QLearningAgent()
    fresh.load(str(path))
    assert fresh.q_table["s"]["a"] == pytest.approx(3.14)
    assert fresh.total_episodes == 5
    assert fresh.victories == 2
