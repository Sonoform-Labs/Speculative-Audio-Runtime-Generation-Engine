import numpy as np

from sage.affordance import AffordanceEngine, GameContext
from sage.fusion import FusedProposal
from sage.generators import MockGenerator
from sage.types import CacheKey


def _engine(**kw):
    base = dict(variants_per_key=2, vram_budget_kb=100_000, gpu_ms_budget=1e9)
    base.update(kw)
    return AffordanceEngine(MockGenerator(), **base)


def test_context_change_keeps_imminent_predicted_key():
    eng = _engine()
    k = CacheKey("footstep", "grass", "normal")
    eng.submit_predictions([FusedProposal(k, 0.5, 0.9, ("motion",))], now=0.0)
    assert k in eng.cache
    eng.on_context_change(GameContext(0.1, None, ["metal"], "room"), now=0.1)
    assert k in eng.cache


def test_context_change_flushes_stale_predicted_key():
    eng = _engine()
    k = CacheKey("footstep", "grass", "normal")
    eng.submit_predictions([FusedProposal(k, 0.5, 0.9, ("motion",))], now=0.0)
    eng.on_context_change(GameContext(1.0, None, ["metal"], "room"), now=1.0)
    assert k not in eng.cache


def test_dead_set_still_flushes_unpredicted_key():
    eng = _engine()
    eng.on_context_change(GameContext(0.0, "gun", ["grass"], "field"))
    assert CacheKey("shoot", "metal", "normal") in eng.cache
    eng.on_context_change(GameContext(5.0, None, ["metal"], "room"))
    assert CacheKey("shoot", "metal", "normal") not in eng.cache


def test_eviction_spares_imminent_under_pressure():
    eng = _engine(variants_per_key=1, vram_budget_kb=40)
    old = CacheKey("thud", "stone", "normal")
    eng.on_event(old)
    eng.drain_demand()
    assert old in eng.cache
    imm = CacheKey("footstep", "grass", "normal")
    eng.submit_predictions([FusedProposal(imm, 0.5, 0.9, ("motion",))], now=0.0)
    assert imm in eng.cache
    assert old not in eng.cache


def test_mispredicted_surface_serves_authored_not_wrong_clip():
    eng = _engine()
    predicted = CacheKey("footstep", "grass", "normal")
    eng.submit_predictions([FusedProposal(predicted, 0.5, 0.9, ("motion",))], now=0.0)
    assert predicted in eng.cache
    served = eng.on_event(CacheKey("footstep", "metal", "normal"))
    assert served == "fallback"
    grass = [c.samples for c in eng.cache[predicted]]
    assert not any(np.array_equal(eng.last_served.samples, g) for g in grass)
