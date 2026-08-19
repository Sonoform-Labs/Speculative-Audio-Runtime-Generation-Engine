from sage.generators import MockGenerator
from sage.metrics import report, summarize
from sage.pipeline import PredictivePipeline
from sage.predictor import Tier1DeterministicPredictor
from sage.types import GameEvent


def _result(lead_ms=300, latency_ms=100, unique=False):
    events = [
        GameEvent("footstep", "wood", "normal", 0.5 * i + 1.0,
                  context=(str(i) if unique else "default"))
        for i in range(20)
    ]
    preds = Tier1DeterministicPredictor(lead_ms=lead_ms).predict(events)
    return PredictivePipeline(MockGenerator(), gen_latency_ms=latency_ms).run(events, preds)


def test_summary_keys_present():
    s = summarize(_result())
    for key in ("hit_rate", "fallback_rate", "ready_in_time_among_predicted",
                "speculation_precision", "cache_peak_kb"):
        assert key in s


def test_hit_rate_perfect_when_lead_exceeds_latency():
    s = summarize(_result(lead_ms=300, latency_ms=100))
    assert s["hit_rate"] == 1.0
    assert s["ready_in_time_among_predicted"] == 1.0


def test_hit_rate_zero_when_latency_exceeds_lead():
    s = summarize(_result(lead_ms=100, latency_ms=500, unique=True))
    assert s["hit_rate"] == 0.0
    assert s["fallback_rate"] == 1.0


def test_report_is_stringy():
    assert "SIMULATION REPORT" in report(_result())
