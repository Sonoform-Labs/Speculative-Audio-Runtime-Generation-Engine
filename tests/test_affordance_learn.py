from sarge.affordance import AffordanceEngine, GameContext
from sarge.affordance_learn import RuntimeAffordanceLearner
from sarge.generators import MockGenerator
from sarge.types import CacheKey


def test_empty_before_observation():
    learner = RuntimeAffordanceLearner()
    assert learner.candidates(GameContext(0.0, None, ["grass"], "field")) == []


def test_generalizes_surface_driven_event_to_new_surface():
    learner = RuntimeAffordanceLearner()
    learner.observe_context(GameContext(0.0, None, ["grass", "metal"], "field"))
    for _ in range(3):
        learner.observe_event(CacheKey("footstep", "grass", "normal"))
        learner.observe_event(CacheKey("footstep", "metal", "normal"))
        learner.observe_event(CacheKey("shoot", "metal", "normal"))
    keys = {str(c.key) for c in learner.candidates(GameContext(1.0, None, ["wood"], "field"))}
    assert "footstep|wood|normal|default" in keys
    assert "shoot|metal|normal|default" in keys


def test_fixed_material_event_keeps_its_material():
    learner = RuntimeAffordanceLearner()
    learner.observe_context(GameContext(0.0, "gun", ["grass"], "field"))
    for _ in range(4):
        learner.observe_event(CacheKey("reload", "metal", "normal"))
    cands = learner.candidates(GameContext(1.0, "gun", ["sand"], "field"))
    keys = {str(c.key) for c in cands}
    assert "reload|metal|normal|default" in keys
    assert "reload|sand|normal|default" not in keys


def test_engine_with_learner_prewarms_a_new_surface_no_tables():
    eng = AffordanceEngine(MockGenerator(), learner=RuntimeAffordanceLearner(),
                           variants_per_key=2, vram_budget_kb=100_000, gpu_ms_budget=1e9)
    eng.on_context_change(GameContext(0.0, None, ["grass", "metal"], "field"))
    for _ in range(3):
        eng.on_event(CacheKey("footstep", "grass", "normal"))
        eng.on_event(CacheKey("footstep", "metal", "normal"))
    eng.on_context_change(GameContext(5.0, None, ["wood"], "field"))
    assert CacheKey("footstep", "wood", "normal") in eng.cache
    assert eng.on_event(CacheKey("footstep", "wood", "normal")) == "cached"
