from sarge.affordance import AffordanceEngine, GameContext
from sarge.generators import MockGenerator
from sarge.types import CacheKey


def _engine():
    return AffordanceEngine(MockGenerator(), variants_per_key=2, target_variants=6,
                            high_rate_variants=8, vram_budget_kb=100_000,
                            gpu_ms_budget=1e9)


def test_high_rate_keys_target_deeper_pools():
    eng = _engine()
    eng.on_context_change(GameContext(0.0, "gun", ["grass"], "field"))
    assert eng.want[CacheKey("footstep", "grass", "normal")] == 8
    assert eng.want[CacheKey("shoot", "metal", "normal")] == 8
    assert eng.want[CacheKey("reload", "metal", "normal")] == 6


def test_deepener_fills_high_rate_to_target():
    eng = _engine()
    eng.on_context_change(GameContext(0.0, None, ["grass"], "field"))
    while eng.deepen_step():
        pass
    assert len(eng.cache[CacheKey("footstep", "grass", "normal")]) == 8


def test_burst_serves_rotate_through_distinct_variants():
    eng = _engine()
    eng.on_context_change(GameContext(0.0, None, ["grass"], "field"))
    while eng.deepen_step():
        pass
    fk = CacheKey("footstep", "grass", "normal")
    seen = []
    for _ in range(8):
        eng.on_event(fk)
        seen.append(eng.last_served.variant_id)
    assert len(set(seen)) == 8


def test_custom_high_rate_set_respected():
    eng = AffordanceEngine(MockGenerator(), target_variants=6, high_rate_variants=10,
                           high_rate_events={"footstep"}, vram_budget_kb=100_000,
                           gpu_ms_budget=1e9)
    eng.on_context_change(GameContext(0.0, "gun", ["grass"], "field"))
    assert eng.want[CacheKey("footstep", "grass", "normal")] == 10
    assert eng.want[CacheKey("shoot", "metal", "normal")] == 6
