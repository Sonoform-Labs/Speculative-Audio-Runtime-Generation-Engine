from predictive_sfx.affordance import AffordanceEngine, GameContext, candidates, make_session
from predictive_sfx.generators import MockGenerator
from predictive_sfx.types import CacheKey


def test_candidates_from_context():
    ctx = GameContext(0.0, "gun", ["metal"], "room")
    keys = {str(c.key) for c in candidates(ctx)}
    assert "shoot|metal|normal|default" in keys
    assert "reload|metal|normal|default" in keys
    assert "footstep|metal|normal|default" in keys


def test_equip_then_fire_is_a_hit():
    eng = AffordanceEngine(MockGenerator())
    eng.on_context_change(GameContext(0.0, "gun", ["grass"], "field"))
    assert eng.on_event(CacheKey("shoot", "metal", "normal")) == "cached"


def test_unanticipated_event_falls_back():
    eng = AffordanceEngine(MockGenerator())
    eng.on_context_change(GameContext(0.0, None, ["grass"], "field"))
    assert eng.on_event(CacheKey("explosion", "metal", "hard")) == "fallback"


def test_dead_set_eviction_on_context_change():
    eng = AffordanceEngine(MockGenerator())
    eng.on_context_change(GameContext(0.0, "gun", ["grass"], "field"))
    assert CacheKey("shoot", "metal", "normal") in eng.cache
    eng.on_context_change(GameContext(5.0, None, ["metal"], "room"))
    assert CacheKey("shoot", "metal", "normal") not in eng.cache


def test_session_mostly_hits():
    eng = AffordanceEngine(MockGenerator())
    for t, kind, payload in make_session():
        eng.on_context_change(payload) if kind == "context" else eng.on_event(payload)
    total = eng.stats["hits"] + eng.stats["fallback"]
    assert eng.stats["hits"] / total >= 0.8
