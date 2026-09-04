from sarge.affordance import AffordanceEngine, GameContext
from sarge.affordance_learn import RuntimeAffordanceLearner
from sarge.generators import MockGenerator
from sarge.router import EventRouter
from sarge.types import CacheKey


def test_router_classifies_by_tolerance():
    r = EventRouter()
    assert r.route("footstep") == "prefetch"
    assert r.route("shoot") == "prefetch"
    assert r.route("ambient") == "realtime"
    assert r.route("music") == "realtime"
    assert r.route("some_unknown_event") == "prefetch"


def test_custom_realtime_set():
    r = EventRouter(realtime_events={"birdsong"})
    assert r.is_realtime("birdsong")
    assert r.is_prefetch("ambient")


def test_engine_router_prewarms_tight_skips_tolerant():
    eng = AffordanceEngine(MockGenerator(), learner=RuntimeAffordanceLearner(), router=EventRouter(),
                           variants_per_key=2, vram_budget_kb=100_000, gpu_ms_budget=1e9)
    eng.on_context_change(GameContext(0.0, None, ["grass", "metal"], "field"))
    for _ in range(3):
        eng.on_event(CacheKey("footstep", "grass", "normal"))
        eng.on_event(CacheKey("footstep", "metal", "normal"))
        eng.on_event(CacheKey("ambient", "forest", "normal"))
    eng.on_context_change(GameContext(5.0, None, ["wood"], "field"))
    assert CacheKey("footstep", "wood", "normal") in eng.cache
    assert not any(k.event_type == "ambient" for k in eng.cache)
