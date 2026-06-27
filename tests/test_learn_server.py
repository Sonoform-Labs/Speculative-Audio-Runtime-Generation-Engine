import pytest

from predictive_sfx.sidecar import serve, SfxClient


@pytest.fixture
def learn_server():
    srv = serve(generator="mock", host="127.0.0.1", port=0, learn=True,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=2)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_serve_learn_wires_learner_and_router(learn_server):
    assert learn_server.engine.learner is not None
    assert learn_server.engine.router is not None


def test_demo_events_train_the_learner_live(learn_server):
    with SfxClient(*learn_server.server_address) as c:
        c.context(0.0, None, ["grass", "metal"], "arena")
        meta = None
        for _ in range(3):
            c.event("footstep", "grass", "normal", "arena")
            c.event("footstep", "metal", "normal", "arena")
            meta, _ = c.event("shoot", "metal", "normal", "arena")
    assert meta["learned"] >= 2  # learned footstep + shoot from play
    assert learn_server.engine.learner.event_count["footstep"] > 0


def test_learner_generalizes_to_new_surface_live(learn_server):
    with SfxClient(*learn_server.server_address) as c:
        c.context(0.0, None, ["grass", "metal"], "arena")
        for _ in range(3):
            c.event("footstep", "grass", "normal", "default")
            c.event("footstep", "metal", "normal", "default")
        c.context(5.0, None, ["wood"], "arena")  # surface never seen before
    eng = learn_server.engine
    keys = set(eng.cache) | eng.demand_set
    assert any(k.event_type == "footstep" and k.material == "wood" for k in keys)
