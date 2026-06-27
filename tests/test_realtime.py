import pytest

from predictive_sfx.affordance import AffordanceEngine, GameContext
from predictive_sfx.generators.mock import MockGenerator
from predictive_sfx.types import CacheKey
from predictive_sfx.sidecar import serve, SfxClient


def _engine():
    return AffordanceEngine(MockGenerator(), realtime=True, vram_budget_kb=200_000)


def test_nothing_generated_at_startup():
    eng = _engine()
    assert len(eng.cache) == 0
    assert eng.stats["spec_gen"] == 0
    assert eng.stats["realtime"] == 0


def test_context_change_pregenerates_nothing():
    eng = _engine()
    eng.on_context_change(GameContext(0.0, "gun", ["grass", "stone", "metal"], "arena"))
    assert len(eng.cache) == 0           # no prewarm
    assert eng.stats["spec_gen"] == 0


def test_miss_generates_in_realtime_then_caches():
    eng = _engine()
    k = CacheKey("footstep", "grass", "normal")
    s1 = eng.on_event(k)
    assert s1 == "realtime"
    assert eng.last_served is not None and eng.last_served.samples.size > 0
    assert k in eng.cache
    assert eng.stats["realtime"] == 1
    s2 = eng.on_event(k)
    assert s2 == "cached"                 # instant from cache on repeat
    assert eng.stats["hits"] == 1
    assert eng.stats["realtime"] == 1     # not regenerated


def test_generalizes_to_an_event_no_table_ever_defined():
    eng = _engine()
    # an event_type + material that appear in no hand-authored table anywhere
    k = CacheKey("teleport_whoosh", "crystal", "hard")
    assert eng.on_event(k) == "realtime"
    assert k in eng.cache
    assert eng.on_event(k) == "cached"


def test_prediction_pregenerates_just_in_time_so_first_event_is_instant():
    from predictive_sfx.fusion import FusedProposal
    eng = _engine()
    k = CacheKey("footstep", "grass", "normal")
    # a live prediction with enough lead time queues a JIT gen job (not on-demand)
    res = eng.register_predictions([FusedProposal(k, eta_s=0.6, p_use=0.9, sources=("motion",))], now=0.0)
    assert res["queued"] == 1 and k in eng.demand_set
    # the background drain (what the deepen loop runs) generates it ahead of the event
    eng.drain_demand(max_jobs=4)
    assert k in eng.cache
    # so when the footstep actually fires, it's an instant cache hit -- never generated on-demand
    assert eng.on_event(k) == "cached"
    assert eng.stats["realtime"] == 0


@pytest.fixture
def realtime_server():
    srv = serve(generator="mock", host="127.0.0.1", port=0, realtime=True, vram_budget_kb=200_000)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_realtime_over_the_wire(realtime_server):
    with SfxClient(*realtime_server.server_address) as c:
        c.context(0.0, "gun", ["grass", "metal"], "arena")  # pre-generates nothing
        m1, pcm1 = c.event("footstep", "grass", "normal", "default")
        m2, _ = c.event("footstep", "grass", "normal", "default")
        m3, _ = c.event("portal_hum", "void", "soft", "default")  # never-defined event
    assert m1["served"] == "realtime" and m1["n_samples"] > 0
    assert m2["served"] == "cached"
    assert m3["served"] == "realtime"
    assert realtime_server.engine.stats["spec_gen"] == 0  # nothing speculative, ever
