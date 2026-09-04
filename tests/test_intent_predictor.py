from sarge.affordance import AffordanceEngine
from sarge.fusion import PredictionFusion
from sarge.generators import MockGenerator
from sarge.predictor import EventSequencePredictor
from sarge.types import CacheKey


def _train_pattern(p, cycles=20):
    for _ in range(cycles):
        for ev in ("shoot", "shoot", "reload"):
            p.observe(ev, "combat")


def test_learns_repeating_pattern():
    p = EventSequencePredictor(order=2)
    _train_pattern(p)
    p.history = ["shoot", "shoot"]
    preds = p.predict("combat")
    assert preds[0][0] == "reload"
    assert preds[0][1] > 0.6


def test_top_k_is_respected():
    p = EventSequencePredictor(order=2, top_k=2)
    for _ in range(10):
        for ev in ("a", "b", "c", "d"):
            p.observe(ev, "x")
    assert len(p.predict("x")) <= 2


def test_backs_off_to_unigram_when_history_unseen():
    p = EventSequencePredictor(order=2, min_total=3)
    for _ in range(10):
        p.observe("shoot", "combat")
    p.history = ["never", "seen"]
    preds = p.predict("combat")
    assert preds and preds[0][0] == "shoot"


def test_unknown_context_returns_empty():
    p = EventSequencePredictor()
    p.observe("shoot", "combat")
    assert p.predict("menu") == []


def test_random_stream_is_not_overconfident():
    import numpy as np
    rng = np.random.default_rng(0)
    p = EventSequencePredictor(order=2)
    evs = ["a", "b", "c", "d", "e"]
    for _ in range(500):
        p.observe(str(rng.choice(evs)), "x")
    top = p.predict("x")
    assert top and top[0][1] < 0.45


def test_intent_predictions_flow_through_fusion_into_engine():
    p = EventSequencePredictor(order=2)
    _train_pattern(p)
    p.history = ["shoot", "shoot"]
    preds = p.predict("combat")
    fusion = PredictionFusion()
    raw = [("intent", CacheKey(ev, "metal", "normal", "combat"), 0.6, prob) for ev, prob in preds]
    proposals = fusion.fuse(raw)
    eng = AffordanceEngine(MockGenerator(), variants_per_key=2, gen_cost_ms=393.0,
                           vram_budget_kb=100_000, gpu_ms_budget=1e9)
    eng.submit_predictions(proposals)
    assert CacheKey("reload", "metal", "normal", "combat") in eng.cache
