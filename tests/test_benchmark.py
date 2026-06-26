from predictive_sfx.benchmark import compare_predictors, footstep_stream, gait_scenario


def test_prediction_beats_no_prediction():
    res = compare_predictors(footstep_stream(n=64), gen_latency_ms=120, lead_ms=300)
    assert res["none"]["hit_rate"] == 0.0
    assert res["oracle"]["warm_hit_rate"] > 0.95
    assert res["none"]["hit_rate"] < res["noisy"]["hit_rate"] < res["oracle"]["hit_rate"] + 1e-9


def test_recurring_keys_are_robust_to_predictor_noise():
    res = compare_predictors(footstep_stream(n=120), gen_latency_ms=120, lead_ms=300)
    assert res["noisy"]["hit_rate"] > 0.95


def test_noisy_predictor_degrades_on_unique_keys():
    res = compare_predictors(footstep_stream(n=120, unique=True), gen_latency_ms=120, lead_ms=300)
    assert 0.3 < res["noisy"]["hit_rate"] < 0.98
    assert res["noisy"]["hit_rate"] < res["oracle"]["hit_rate"]


def test_cold_start_is_worse_than_warm():
    res = compare_predictors(footstep_stream(n=96, surfaces=("grass", "metal", "wood")))
    assert res["oracle"]["cold_hit_rate"] <= res["oracle"]["warm_hit_rate"]


def test_gait_scenario_predicts_most_steps():
    s = gait_scenario(speed=1.4, dur=12.0)
    assert s["n_steps"] >= 8
    assert s["predicted"]["hit_rate"] > s["none"]["hit_rate"]
    assert s["predicted"]["warm_hit_rate"] > 0.7
