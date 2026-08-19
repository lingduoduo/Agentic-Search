import pytest

from src.model.pre_training.intents.metrics import (
    IntentPredictionRecord,
    ModulePredictionRecord,
    module_metrics_report,
    realistic_accuracy_report,
    separability_report,
)


def test_realistic_accuracy_reports_argmax_accuracy_and_covered_accuracy():
    records = [
        IntentPredictionRecord("e1", "search", "search", 0.80, 1.0, "model"),
        IntentPredictionRecord("e2", "chat", "search", 0.40, 1.0, "model"),
        IntentPredictionRecord("e3", "tool", "tool", 0.90, 1.0, "model"),
    ]

    report = realistic_accuracy_report(records, threshold=0.75)

    assert report["total_queries"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["covered_accuracy"] == pytest.approx(1.0)
    assert set(report["per_label_metrics"]) == {"chat", "search", "tool"}


def test_module_macro_f1_excludes_the_form_label():
    """bare_entity describes utterance shape, not intent; it would distort F1."""

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        ModulePredictionRecord("b", ("bare_entity",), ("lookup_fact",), True),
    ]

    report = module_metrics_report(records)

    assert "bare_entity" not in report["per_module_metrics"]
    # b's predicted lookup_fact has no matching gold lookup_fact (b's gold is
    # bare_entity, excluded), so it is a false positive against lookup_fact:
    # precision 1/2. This exercises the false-positive path; recall would be
    # 1.0 here regardless of whether bare_entity were excluded.
    assert report["per_module_metrics"]["lookup_fact"]["precision"] == pytest.approx(
        0.5
    )


def test_macro_f1_divides_by_every_semantic_module_not_only_those_that_appear():
    """The denominator is fixed at len(SEMANTIC_MODULES) == 13, always."""
    from src.model.pre_training.intents.model import SEMANTIC_MODULES

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        ModulePredictionRecord("b", ("explain",), ("explain",), True),
        ModulePredictionRecord("c", ("create",), ("create",), True),
    ]

    report = module_metrics_report(records)

    # Perfect (f1=1.0) on the 3 modules that appear; the other 10 semantic
    # modules have no support and no predictions, so f1=0.0 for each, yet
    # they still count in the average's denominator.
    assert report["macro_f1"] == pytest.approx(3 / len(SEMANTIC_MODULES))


def test_joint_accuracy_requires_route_and_exact_module_set():
    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        # right route, extra module -> not joint-correct
        ModulePredictionRecord(
            "b", ("lookup_fact",), ("lookup_fact", "current_info"), True
        ),
        # right modules, wrong route -> not joint-correct
        ModulePredictionRecord("c", ("explain",), ("explain",), False),
    ]

    report = module_metrics_report(records)

    assert report["joint_accuracy"] == pytest.approx(1 / 3)


def test_module_set_order_does_not_affect_joint_accuracy():
    records = [
        ModulePredictionRecord(
            "a", ("lookup_fact", "current_info"), ("current_info", "lookup_fact"), True
        )
    ]

    assert module_metrics_report(records)["joint_accuracy"] == pytest.approx(1.0)


def test_module_report_records_how_many_queries_carried_gold_modules():
    """The legacy 30 predate modules; the metric must say what it covered."""

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        ModulePredictionRecord("b", (), ("explain",), True),
    ]

    report = module_metrics_report(records)

    assert report["scored_queries"] == 1
    assert report["total_queries"] == 2


def test_separability_auc_is_one_for_perfectly_separated_scores():
    report = separability_report([0.8, 0.9, 0.85], [0.1, 0.2, 0.15])

    assert report["auc"] == pytest.approx(1.0)
    assert report["cohens_d"] > 0


def test_separability_auc_is_half_for_identical_distributions():
    report = separability_report([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])

    assert report["auc"] == pytest.approx(0.5)


def test_raw_margin_can_shrink_while_separability_improves():
    """The whole reason the bar changed units: raw margin is scale-dependent."""

    wide = separability_report([0.9, 0.5], [0.4, 0.0])  # margin 0.5
    tight = separability_report([0.81, 0.80], [0.79, 0.78])  # margin 0.02

    assert tight["raw_margin"] < wide["raw_margin"]
    assert tight["auc"] >= wide["auc"]


def test_separability_reports_the_overlap_bounds():
    report = separability_report([0.4, 0.9], [0.1, 0.5])

    assert report["max_out_of_scope"] == pytest.approx(0.5)
    assert report["min_in_scope"] == pytest.approx(0.4)


def test_separability_rejects_an_empty_group():
    with pytest.raises(ValueError, match="empty"):
        separability_report([], [0.1])
