from predictive_sfx.affordance import AffordanceEngine, GameContext
from predictive_sfx.affordance_learn import RuntimeAffordanceLearner
from predictive_sfx.generators import MockGenerator
from predictive_sfx.router import EventRouter
from predictive_sfx.types import CacheKey


def test_router_classifies_by_tolerance():
    r = EventRouter()
    assert r.route("footstep") == "prefetch"
    assert r.route("shoot") == "prefetch"
    assert r.route("ambient") == "realtime"
    assert r.route("music") == "realtime"
    assert r.route("some_unknown_event") == "prefetch"  # default safe


def test_custom_realtime_set():
    r = EventRouter(realtime_events={"birdsong"})
    assert r.is_realtime("birdsong")
    assert r.is_prefetch("ambient")  # not in custom set -> prefetch


def test_engine_router_prewarms_tight_skips_tolerant():
    eng = AffordanceEngine(MockGenerator(), learner=RuntimeAffordanceLearner(), router=EventRouter(),
                           variants_per_key=2, vram_budget_kb=100_000, gpu_ms_budget=1e9)
    eng.on_context_change(GameContext(0.0, None, ["grass", "metal"], "field"))
    for _ in range(3):
        eng.on_event(CacheKey("footstep", "grass", "normal"))
        eng.on_event(CacheKey("footstep", "metal", "normal"))
        eng.on_event(CacheKey("ambient", "forest", "normal"))
    eng.on_context_change(GameContext(5.0, None, ["wood"], "field"))
    assert CacheKey("footstep", "wood", "normal") in eng.cache  # tight -> prefetched
    assert not any(k.event_type == "ambient" for k in eng.cache)  # tolerant -> on-demand, not prewarmed
