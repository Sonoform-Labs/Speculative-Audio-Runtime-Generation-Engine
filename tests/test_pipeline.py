from sarge.generators import MockGenerator
from sarge.pipeline import PredictivePipeline
from sarge.predictor import Tier1DeterministicPredictor
from sarge.types import CacheKey, GameEvent


def _events(n=30):
    return [GameEvent("footstep", "wood", "normal", 0.5 * i + 1.0) for i in range(n)]


def test_hit_when_lead_exceeds_latency():
    events = _events()
    preds = Tier1DeterministicPredictor(lead_ms=300).predict(events)
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=100)
    res = pipe.run(events, preds)
    assert all(o.hit for o in res.outcomes)
    assert all(o.served == "cached" for o in res.outcomes)


def test_miss_falls_back_when_latency_exceeds_lead():
    events = [GameEvent("footstep", "wood", "normal", 0.5 * i + 1.0, context=str(i))
              for i in range(20)]
    preds = Tier1DeterministicPredictor(lead_ms=100).predict(events)
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=500)
    res = pipe.run(events, preds)
    assert not any(o.hit for o in res.outcomes)
    assert all(o.served == "fallback" for o in res.outcomes)


def test_warm_cache_reuse_serves_recurring_keys():
    events = _events(10)
    preds = Tier1DeterministicPredictor(lead_ms=100).predict(events)
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=300)
    res = pipe.run(events, preds)
    assert res.outcomes[0].served == "fallback"
    assert all(o.hit for o in res.outcomes[1:])


def test_prebake_guarantees_hit_regardless_of_latency():
    events = _events()
    preds = Tier1DeterministicPredictor(lead_ms=0).predict(events)
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=5000)
    pipe.prebake({CacheKey("footstep", "wood", "normal")}, variants=4, pin=True)
    res = pipe.run(events, preds)
    assert all(o.hit and o.from_prebake for o in res.outcomes)


def test_silence_when_no_authored_fallback():
    events = [GameEvent("footstep", "wood", "normal", 0.5 * i + 1.0, context=str(i))
              for i in range(5)]
    preds = Tier1DeterministicPredictor(lead_ms=100).predict(events)
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=500, authored_assets=set())
    res = pipe.run(events, preds)
    assert all(o.served == "silence" for o in res.outcomes)


def test_wasted_generations_counted():
    events = _events(10)
    from sarge.predictor import Prediction
    preds = [Prediction(CacheKey("footstep", "metal", "soft"), 0.0, 0.5)]
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=100)
    res = pipe.run(events, preds)
    assert res.n_wasted_generations == 1
