import numpy as np

from sarge.predictor import KinematicGaitPredictor


def _straight_walk(speed=1.5, dur=3.0, dt=0.05, surface="grass"):
    n = int(dur / dt)
    return [(i * dt, (speed * i * dt, 0.0, 0.0), surface) for i in range(n)]


def test_straight_walk_emits_regular_footsteps():
    pred = KinematicGaitPredictor(stride_m=0.75, lead_ms=300, min_speed=0.4)
    preds = pred.run(_straight_walk(speed=1.5, dur=3.0))
    assert len(preds) >= 4
    assert all(p.key.event_type == "footstep" and p.key.material == "grass" for p in preds)
    targets = [p.target_time_s for p in preds]
    gaps = np.diff(targets)
    expected = 0.75 / 1.5
    assert np.allclose(gaps, expected, atol=0.08)


def test_lead_time_is_within_budget():
    pred = KinematicGaitPredictor(stride_m=0.75, lead_ms=300, min_speed=0.4)
    preds = pred.run(_straight_walk(speed=1.5, dur=3.0))
    leads = [p.lead_s for p in preds]
    assert all(0.0 < lead <= 0.3 + 1e-6 for lead in leads)
    assert max(leads) > 0.15


def test_standing_still_emits_nothing():
    pred = KinematicGaitPredictor(min_speed=0.4)
    samples = [(i * 0.05, (0.001 * i, 0.0, 0.0), "grass") for i in range(40)]
    assert pred.run(samples) == []


def test_straight_walk_is_high_confidence():
    pred = KinematicGaitPredictor(stride_m=0.75, lead_ms=300)
    preds = pred.run(_straight_walk(speed=1.5, dur=3.0))
    assert all(p.confidence > 0.95 for p in preds)


def test_sustained_turn_lowers_confidence_vs_straight():
    import math
    dt, speed = 0.05, 1.5
    straight = KinematicGaitPredictor(stride_m=0.75, lead_ms=300).run(_straight_walk(speed, 4.0, dt))
    samples, x, y, t, ang = [], 0.0, 0.0, 0.0, 0.0
    for _ in range(120):
        samples.append((t, (x, y, 0.0), "grass"))
        ang += 0.8
        x += speed * dt * math.cos(ang)
        y += speed * dt * math.sin(ang)
        t += dt
    turning = KinematicGaitPredictor(stride_m=0.75, lead_ms=300).run(samples)
    assert min(p.confidence for p in straight) > 0.95
    assert turning and min(p.confidence for p in turning) < 0.85
    assert min(p.confidence for p in turning) < min(p.confidence for p in straight) - 0.1


def test_upcoming_surface_overrides_current():
    pred = KinematicGaitPredictor(stride_m=0.75, lead_ms=300)
    samples = [(i * 0.05, (1.5 * i * 0.05, 0.0, 0.0), "grass", "metal") for i in range(60)]
    preds = pred.run(samples)
    assert preds and all(p.key.material == "metal" for p in preds)
