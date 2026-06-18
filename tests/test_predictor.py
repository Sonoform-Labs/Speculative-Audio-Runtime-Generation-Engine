from predictive_sfx.predictor import NoisyPredictor, Tier1DeterministicPredictor
from predictive_sfx.types import GameEvent


def _events(n=50):
    return [GameEvent("footstep", "wood", "normal", 0.5 * i + 0.5) for i in range(n)]


def test_tier1_perfect_key_and_fixed_lead():
    events = _events(20)
    preds = Tier1DeterministicPredictor(lead_ms=250).predict(events)
    assert len(preds) == len(events)
    for e, p in zip(events, preds):
        assert p.key == e.key
        assert abs(p.lead_s - 0.25) < 1e-9


def test_tier1_clamps_lead_at_zero():
    events = [GameEvent("footstep", "wood", "normal", 0.1)]
    p = Tier1DeterministicPredictor(lead_ms=500).predict(events)[0]
    assert p.emit_time_s == 0.0


def test_noisy_recall_bounds_prediction_count():
    events = _events(200)
    preds = NoisyPredictor(recall=0.5, key_accuracy=1.0, false_pos_rate=0.0, seed=1).predict(events)
    # roughly half predicted; allow generous slack for randomness
    assert 70 <= len(preds) <= 130


def test_noisy_false_positives_create_wrong_keys():
    events = _events(100)
    preds = NoisyPredictor(recall=1.0, key_accuracy=1.0, false_pos_rate=1.0, seed=2).predict(events)
    # every event yields its correct prediction plus one wasted guess
    wrong = [p for p in preds if p.key != events[0].key]
    assert len(preds) > len(events)
    assert len(wrong) > 0
