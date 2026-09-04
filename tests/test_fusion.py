from sarge.affordance import AffordanceEngine
from sarge.calibration import PredictorCalibration
from sarge.fusion import PredictionFusion
from sarge.generators import MockGenerator
from sarge.types import CacheKey


def test_duplicate_proposals_combine_noisy_or():
    f = PredictionFusion()
    k = CacheKey("footstep", "grass", "normal")
    out = f.fuse([("motion", k, 0.5, 0.5), ("affordance", k, 0.5, 0.5)])
    assert len(out) == 1
    assert abs(out[0].p_use - 0.75) < 1e-9
    assert set(out[0].sources) == {"motion", "affordance"}


def test_owner_wins_eta():
    f = PredictionFusion(owner={"footstep": "motion"})
    k = CacheKey("footstep", "grass", "normal")
    out = f.fuse([("affordance", k, 0.1, 0.5), ("motion", k, 0.3, 0.5)])
    assert abs(out[0].eta_s - 0.3) < 1e-9


def test_min_ev_threshold_filters_low_value():
    f = PredictionFusion(min_ev=0.5, gen_cost_ms=400, latency_saved_ms=400)
    k = CacheKey("shoot", "metal", "normal")
    low = f.fuse([("intent", k, 0.5, 0.2)])
    assert low == []
    high = f.fuse([("intent", k, 0.5, 0.9)])
    assert len(high) == 1


def test_calibration_pulls_overconfident_source_down():
    import numpy as np
    rng = np.random.default_rng(0)
    cal = PredictorCalibration(min_samples=20)
    for _ in range(2000):
        cal.observe("intent", 0.9, rng.random() < 0.2)
    f = PredictionFusion(calibration=cal)
    k = CacheKey("shoot", "metal", "normal")
    out = f.fuse([("intent", k, 0.5, 0.9)])
    assert out[0].p_use < 0.35


def test_ranking_is_by_expected_value():
    f = PredictionFusion()
    a = CacheKey("a", "m", "normal")
    b = CacheKey("b", "m", "normal")
    out = f.fuse([("s", a, 0.5, 0.2), ("s", b, 0.5, 0.9)])
    assert out[0].key == b and out[1].key == a


def test_submit_predictions_generates_in_budget_and_gates_late():
    eng = AffordanceEngine(MockGenerator(), variants_per_key=2, gen_cost_ms=393.0,
                           vram_budget_kb=100_000, gpu_ms_budget=1e9)
    f = PredictionFusion()
    good = CacheKey("footstep", "grass", "normal")
    late = CacheKey("footstep", "metal", "normal")
    proposals = f.fuse([("motion", good, 0.5, 0.9), ("motion", late, 0.05, 0.9)])
    stats = eng.submit_predictions(proposals)
    assert good in eng.cache
    assert late not in eng.cache
    assert stats["lead_gated"] == 1
    assert eng.on_event(good) == "cached"
