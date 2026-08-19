from sage.gamestate import AnticipatoryPredictor, simulate


def test_simulate_produces_frames_and_events():
    tr = simulate(seed=0, duration_s=20.0)
    assert len(tr.frames) > 1000
    assert len(tr.events) > 10
    assert all(e.event_type == "footstep" for e in tr.events)


def test_predictor_is_causal():
    tr = simulate(seed=1, duration_s=30.0)
    preds = AnticipatoryPredictor(horizon_ms=350).predict(tr)
    assert len(preds) > 10
    assert all(p.emit_time_s <= p.target_time_s for p in preds)
    assert len(preds) <= 2 * len(tr.events)


def test_predictor_forecasts_not_peeks():
    tr = simulate(seed=2, duration_s=20.0)
    preds = AnticipatoryPredictor().predict(tr)
    ev_times = {round(e.time_s, 4) for e in tr.events}
    exact = sum(1 for p in preds if round(p.target_time_s, 4) in ev_times)
    assert exact < len(preds)
