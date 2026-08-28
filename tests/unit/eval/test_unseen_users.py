"""Contracts for the split, the measurements, and the report."""

from __future__ import annotations

import pytest

from src.model.post_training.eval.cohort import (
    CohortConfig,
    EvalRecord,
    generate_cohort,
)
from src.model.post_training.eval.unseen_users import (
    BEHAVIOR_COMPONENTS,
    evaluate_unseen_users,
    format_report,
    split_users,
)

from src.model.post_training.eval.instruction_following import CONSTRAINT_NAMES
from src.model.post_training.eval.stats import benjamini_hochberg


def test_split_is_deterministic_for_a_fixed_seed():
    ids = [f"u{i}" for i in range(50)]

    assert split_users(ids, seed=4) == split_users(ids, seed=4)


def test_split_is_independent_of_input_order():
    ids = [f"u{i}" for i in range(50)]
    forward = split_users(ids, seed=4)
    backward = split_users(list(reversed(ids)), seed=4)

    assert forward == backward


def test_no_user_appears_on_both_sides():
    train, holdout = split_users([f"u{i}" for i in range(80)], seed=1)

    assert not (train & holdout)


def test_split_covers_every_user():
    ids = [f"u{i}" for i in range(80)]
    train, holdout = split_users(ids, seed=1)

    assert train | holdout == set(ids)


def test_holdout_fraction_is_approximately_respected():
    train, holdout = split_users(
        [f"u{i}" for i in range(400)], holdout_fraction=0.25, seed=2
    )

    assert 0.18 <= len(holdout) / 400 <= 0.32


def test_a_different_seed_moves_users_across_the_split():
    _, first = split_users([f"u{i}" for i in range(200)], seed=1)
    _, second = split_users([f"u{i}" for i in range(200)], seed=2)

    assert first != second


def test_report_measures_only_held_out_users():
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=8, seed=3))
    _, holdout = split_users({r.user_id for r in records}, seed=3)

    report = evaluate_unseen_users(records, seed=3, resamples=200)

    assert report.n_holdout_users == len(holdout)


def test_a_train_only_users_metric_never_reaches_the_report():
    """Unlike the count check above, this depends on the frame actually being
    filtered: a train-only user with a wildly different metric value must not
    move a held-out user's reported mean at all."""
    seed = 7
    candidates = [f"fu{i}" for i in range(60)]
    train, holdout = split_users(candidates, seed=seed)
    held_user = sorted(holdout)[0]
    train_user = sorted(train)[0]

    records = []
    for index in range(5):
        for policy in ("trained", "baseline"):
            records.append(
                EvalRecord(
                    user_id=held_user,
                    prompt_id=f"p{index}",
                    policy=policy,
                    reward=0.0,
                    converted=True,
                    response="",
                    metrics={"search_rounds": 1.0},
                )
            )
            records.append(
                EvalRecord(
                    user_id=train_user,
                    prompt_id=f"p{index}",
                    policy=policy,
                    reward=0.0,
                    converted=True,
                    response="",
                    metrics={"search_rounds": 999.0},
                )
            )

    report = evaluate_unseen_users(records, seed=seed, resamples=50)
    rounds = next(r for r in report.behavior if r.name == "search_rounds")

    assert rounds.trained_mean == pytest.approx(1.0)
    assert rounds.baseline_mean == pytest.approx(1.0)


def test_alignment_is_detected_on_an_aligned_cohort():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, alignment=3.0, seed=10)
    )

    report = evaluate_unseen_users(records, seed=10, resamples=400)

    assert report.alignment.auc > 0.5
    assert report.alignment.ci_low > 0.5


def test_alignment_reports_how_many_users_it_dropped():
    records = generate_cohort(CohortConfig(num_users=40, sessions_per_user=10, seed=11))

    report = evaluate_unseen_users(records, seed=11, resamples=200)

    assert report.alignment.n_excluded >= 0
    assert report.alignment.n_users + report.alignment.n_excluded == (
        report.n_holdout_users
    )


def test_behavioral_separation_is_detected_when_planted():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, behavior_shift=2.5, seed=12)
    )

    report = evaluate_unseen_users(records, seed=12, resamples=400)
    rounds = next(r for r in report.behavior if r.name == "search_rounds")

    assert rounds.trained_mean < rounds.baseline_mean
    assert rounds.p_adjusted < 0.05
    assert rounds.effect < 0.0


def test_instruction_following_beats_the_baseline_when_planted():
    records = generate_cohort(
        CohortConfig(num_users=60, sessions_per_user=12, instruction_gap=0.35, seed=13)
    )

    report = evaluate_unseen_users(records, seed=13, resamples=400)
    tags = next(r for r in report.instruction if r.name == "answer_tag_present")

    assert tags.trained_mean > tags.baseline_mean
    assert tags.p_adjusted < 0.05


def test_every_component_and_constraint_is_reported():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=14))

    report = evaluate_unseen_users(records, seed=14, resamples=200)

    assert {r.name for r in report.behavior} == set(BEHAVIOR_COMPONENTS)
    assert {r.name for r in report.instruction} == set(CONSTRAINT_NAMES)


def test_adjusted_p_values_are_never_below_raw():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=15))

    report = evaluate_unseen_users(records, seed=15, resamples=200)

    for result in (*report.behavior, *report.instruction):
        assert result.p_adjusted >= result.p_value


def test_evaluation_is_deterministic():
    records = generate_cohort(CohortConfig(num_users=25, sessions_per_user=6, seed=16))

    first = evaluate_unseen_users(records, seed=16, resamples=200)
    second = evaluate_unseen_users(records, seed=16, resamples=200)

    assert first == second


def test_evaluation_rejects_an_empty_record_set():
    with pytest.raises(ValueError, match="records"):
        evaluate_unseen_users([], seed=0)


def test_formatted_report_states_its_provenance_and_user_count():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=17))

    report = evaluate_unseen_users(
        records, seed=17, resamples=200, provenance="simulated cohort"
    )
    text = format_report(report)

    assert "simulated cohort" in text
    assert str(report.n_holdout_users) in text
    assert "held-out users" in text


def test_formatted_report_shows_effect_size_beside_every_p_value():
    records = generate_cohort(CohortConfig(num_users=30, sessions_per_user=8, seed=18))

    text = format_report(evaluate_unseen_users(records, seed=18, resamples=200))

    for name in (*BEHAVIOR_COMPONENTS, *CONSTRAINT_NAMES):
        assert name in text
    assert "effect" in text.lower()


def test_a_high_session_user_cannot_outweigh_low_session_users():
    """Pins the property the plan calls the single most important one in the
    branch: metrics collapse to one value per user before any test runs, so a
    user's session count must not weight their influence on the reported
    mean. One user contributes 100 sessions at trained=10/baseline=0; another
    contributes 2 sessions with no trained/baseline difference at all. Session
    pooling would drag the mean toward ~9.8 (100 heavy sessions swamping 2
    light ones); per-user collapse must average the two users' own means to
    exactly 5.0.
    """
    seed = 42
    candidates = [f"cu{i}" for i in range(40)]
    _, holdout = split_users(candidates, seed=seed)
    holdout_ids = sorted(holdout)
    assert len(holdout_ids) >= 2
    heavy_user, light_user = holdout_ids[0], holdout_ids[1]

    records = []
    for index in range(100):
        for policy, value in (("trained", 10.0), ("baseline", 0.0)):
            records.append(
                EvalRecord(
                    user_id=heavy_user,
                    prompt_id=f"p{index}",
                    policy=policy,
                    reward=0.0,
                    converted=True,
                    response="",
                    metrics={"search_rounds": value},
                )
            )
    for index in range(2):
        for policy in ("trained", "baseline"):
            records.append(
                EvalRecord(
                    user_id=light_user,
                    prompt_id=f"p{index}",
                    policy=policy,
                    reward=0.0,
                    converted=True,
                    response="",
                    metrics={"search_rounds": 0.0},
                )
            )

    report = evaluate_unseen_users(records, seed=seed, resamples=50)
    rounds = next(r for r in report.behavior if r.name == "search_rounds")

    assert rounds.trained_mean == pytest.approx(5.0)
    assert rounds.trained_mean < 9.0  # session-pooled would land near 9.8


def test_bh_correction_is_applied_once_over_the_full_nine_member_family():
    """Pins that the correction is BH over the combined 9-member family (5
    behavior + 4 instruction), not skipped and not computed as two separate
    5- and 4-member corrections. Recomputes ``benjamini_hochberg`` directly on
    the extracted raw p-values, matched by name so a pairing/ordering bug is
    also caught.
    """
    records = generate_cohort(
        CohortConfig(
            num_users=60,
            sessions_per_user=12,
            behavior_shift=2.5,
            instruction_gap=0.35,
            seed=99,
        )
    )
    report = evaluate_unseen_users(records, seed=99, resamples=300)

    family = [*report.behavior, *report.instruction]
    expected_by_name = dict(
        zip(
            [result.name for result in family],
            benjamini_hochberg([result.p_value for result in family]),
        )
    )
    for result in family:
        assert result.p_adjusted == pytest.approx(expected_by_name[result.name])

    # A per-group (5-alone, 4-alone) correction is a different, wrong
    # computation; at least one member must disagree with it.
    per_group_by_name = dict(
        zip(
            [result.name for result in report.behavior],
            benjamini_hochberg([result.p_value for result in report.behavior]),
        )
    )
    per_group_by_name.update(
        zip(
            [result.name for result in report.instruction],
            benjamini_hochberg([result.p_value for result in report.instruction]),
        )
    )
    assert any(
        result.p_adjusted != pytest.approx(per_group_by_name[result.name])
        for result in family
    )
