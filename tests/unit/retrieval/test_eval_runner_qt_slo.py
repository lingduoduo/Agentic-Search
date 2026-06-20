from src.internal.retrieval.eval_runner import qt_slo_exceeded


def test_p99_within_budget():
    assert qt_slo_exceeded([10.0] * 100, slo_ms=50) is False


def test_p99_exceeds_budget():
    lat = [10.0] * 99 + [500.0]
    assert qt_slo_exceeded(lat, slo_ms=50) is True


def test_empty_latencies_never_exceed():
    assert qt_slo_exceeded([], slo_ms=50) is False
