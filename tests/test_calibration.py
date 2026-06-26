import numpy as np

from predictive_sfx.calibration import Calibrator, PredictorCalibration


def test_overconfident_predictor_is_pulled_down():
    rng = np.random.default_rng(0)
    c = Calibrator(min_samples=20)
    for _ in range(2000):
        c.observe(0.9, rng.random() < 0.3)
    assert abs(c.calibrate(0.9) - 0.3) < 0.05


def test_well_calibrated_stays_put():
    rng = np.random.default_rng(1)
    c = Calibrator()
    for _ in range(8000):
        raw = float(rng.random())
        c.observe(raw, rng.random() < raw)
    for p in (0.25, 0.55, 0.85):
        assert abs(c.calibrate(p) - p) < 0.08


def test_sparse_data_returns_raw():
    c = Calibrator(min_samples=20)
    c.observe(0.5, True)
    assert c.calibrate(0.7) == 0.7


def test_hit_rate_tracked():
    c = Calibrator()
    for _ in range(10):
        c.observe(0.5, True)
    for _ in range(10):
        c.observe(0.5, False)
    assert abs(c.hit_rate - 0.5) < 1e-9


def test_per_source_calibration_is_independent():
    pc = PredictorCalibration(min_samples=20)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        pc.observe("motion", 0.9, True)
        pc.observe("intent", 0.9, rng.random() < 0.2)
    assert pc.calibrate("motion", 0.9) > 0.85
    assert pc.calibrate("intent", 0.9) < 0.35
    assert pc.hit_rate("intent") < pc.hit_rate("motion")


def test_calibration_is_monotonic_for_staircase_predictor():
    rng = np.random.default_rng(3)
    c = Calibrator()
    for _ in range(8000):
        raw = float(rng.choice([0.15, 0.45, 0.75]))
        true_p = {0.15: 0.1, 0.45: 0.5, 0.75: 0.9}[raw]
        c.observe(raw, rng.random() < true_p)
    assert c.calibrate(0.15) < c.calibrate(0.45) < c.calibrate(0.75)
